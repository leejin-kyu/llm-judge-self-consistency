# -*- coding: utf-8 -*-
"""Provider-agnostic LLM judge client (OpenAI default, Anthropic optional) + disk cache.

- verdict만 사용(A / B / tie). logprob 불필요 → MPCJ의 training-free 불확실성과 일치.
- 디스크 캐시: 동일 (model,temp,prompt) 호출 재사용 → 메서드 간/재실행 시 실제 API 비용 절감.
  · judge.calls       = 논리 호출 수(메서드 본연 비용; 논문 cost 지표)
  · judge.api_calls   = 실제 과금 호출 수(캐시 미스)
"""
from __future__ import annotations
import os, re, time, json, hashlib, threading

try:  # load code/.env so every runner gets the API key without its own boilerplate
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except ImportError:
    pass

PAIRWISE_SYS = (
    "You are a careful, impartial evaluator. Judge ONLY the specified criterion. "
    "Ignore response length and position. Output exactly one token: A, B, or tie."
)
PAIRWISE_USER = """[Instruction]
{x}

[Response A]
{a}

[Response B]
{b}

[Criterion] {criterion}: {criterion_desc}
Which response is better on this criterion alone? Answer with exactly: A, B, or tie."""

# G-Eval-style holistic CoT prompt (baseline)
COT_SYS = ("You are an impartial evaluator. Think step by step about which response better "
           "follows the instruction, considering correctness, completeness, and relevance; "
           "ignore length and position. End with a final line 'Verdict: A', 'Verdict: B', or 'Verdict: tie'.")
COT_USER = PAIRWISE_USER.replace("[Criterion] {criterion}: {criterion_desc}\n", "")

# criterion-focused CoT prompt (for MPCJ-CoT)
COT_CRIT_SYS = ("You are an impartial evaluator. Briefly reason about which response is better "
                "ON THE GIVEN CRITERION ONLY; ignore length and position. "
                "End with a final line 'Verdict: A', 'Verdict: B', or 'Verdict: tie'.")

CRITERIA = {
    "correctness":  "factual/logical correctness of the answer",
    "completeness": "covers the necessary points of the instruction",
    "relevance":    "stays on-topic and addresses the instruction",
    "conciseness":  "appropriately concise; not padded or verbose",
    "faithfulness": "no fabrication; consistent with given context",
}

_CACHE_LOCK = threading.Lock()
_CACHES: dict[str, dict] = {}


def _load_cache(path):
    if path in _CACHES:
        return _CACHES[path]
    c = {}
    if path and os.path.exists(path):
        try:
            c = json.load(open(path, encoding="utf-8"))
        except Exception:
            c = {}
    _CACHES[path] = c
    return c


def _save_cache(path):
    if not path:
        return
    with _CACHE_LOCK:  # snapshot under the same lock that guards mutation
        snapshot = dict(_CACHES.get(path, {}))
    json.dump(snapshot, open(path, "w", encoding="utf-8"))


def _parse_verdict_status(text: str) -> tuple[str, str]:
    """Return (verdict, parse_status). Status separates parsing failures from
    genuine tie verdicts (P1-15): verdict_line | token | fallback_tie."""
    t = (text or "").strip().lower()
    m = re.search(r"verdict:\s*([ab]|tie)", t)
    if m:
        v = "tie" if m.group(1) == "tie" else m.group(1).upper()
        return v, "verdict_line"
    if re.search(r"\btie\b", t):
        return "tie", "token"
    m = re.search(r"\b([ab])\b", t)
    if m:
        return m.group(1).upper(), "token"
    if t[:1] in ("a", "b"):
        return t[:1].upper(), "token"
    return "tie", "fallback_tie"


def _parse_verdict(text: str) -> str:
    return _parse_verdict_status(text)[0]


class JudgeConfig:
    def __init__(self, provider=None, model=None, temperature=0.0, max_retries=4,
                 cache_path="judge_cache.json", cache_stochastic=False, base_url=None):
        self.provider = provider or os.getenv("MPCJ_JUDGE_PROVIDER", "openai")
        self.model = model or os.getenv("MPCJ_JUDGE_MODEL", "gpt-4o")
        self.temperature = temperature
        self.max_retries = max_retries
        self.cache_path = cache_path
        self.cache_stochastic = cache_stochastic  # True reproduces the caching pitfall (#12)
        self.base_url = base_url  # e.g. http://localhost:11434/v1 for Ollama (OpenAI-compatible)


class Judge:
    def __init__(self, cfg: JudgeConfig | None = None):
        self.cfg = cfg or JudgeConfig()
        self._client = None
        self.calls = 0       # logical
        self.api_calls = 0   # billed (cache miss)
        self.tokens = 0
        self.cache = _load_cache(self.cfg.cache_path)
        # per-call instrumentation (P2-19/P1-15). NOTE: one Judge instance per
        # worker thread — self.log/last-entry updates are not shared-safe.
        self.log: list[dict] = []
        self._miss_since_save = 0

    def _ensure(self):
        if self._client is not None:
            return
        if self.cfg.provider == "openai":
            from openai import OpenAI
            if self.cfg.base_url:
                self._client = OpenAI(base_url=self.cfg.base_url, api_key="ollama")
            else:
                self._client = OpenAI()
        elif self.cfg.provider == "anthropic":
            import anthropic
            self._client = anthropic.Anthropic()
        else:
            raise ValueError(f"unknown provider {self.cfg.provider}")

    def _key(self, system, user, temperature):
        raw = f"{self.cfg.provider}|{self.cfg.model}|{temperature}|{system}|{user}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def _chat(self, system, user, temperature, max_tokens=8):
        self.calls += 1
        # ⚠️ 결함수정: temperature>0(자기일관성 샘플링)은 캐시하면 동일출력으로 붕괴 → 캐시 우회.
        # (cache_stochastic=True이면 결함을 의도적으로 재현 — #12 실험용)
        use_cache = (temperature == 0.0) or self.cfg.cache_stochastic
        key = self._key(system, user, temperature) if use_cache else None
        if use_cache and key in self.cache:
            self.log.append({"temp": temperature, "cached": True, "latency": 0.0,
                             "in_tok": 0, "out_tok": 0, "parse": None})
            return self.cache[key]
        self._ensure()
        last = None
        for attempt in range(self.cfg.max_retries):
            try:
                t0 = time.time()
                in_tok = out_tok = 0
                if self.cfg.provider == "openai":
                    r = self._client.chat.completions.create(
                        model=self.cfg.model, temperature=temperature, max_tokens=max_tokens,
                        messages=[{"role": "system", "content": system},
                                  {"role": "user", "content": user}])
                    out = r.choices[0].message.content
                    if r.usage:
                        self.tokens += r.usage.total_tokens
                        in_tok = getattr(r.usage, "prompt_tokens", 0) or 0
                        out_tok = getattr(r.usage, "completion_tokens", 0) or 0
                else:
                    r = self._client.messages.create(
                        model=self.cfg.model, temperature=temperature, max_tokens=max_tokens,
                        system=system, messages=[{"role": "user", "content": user}])
                    out = r.content[0].text
                    if getattr(r, "usage", None):
                        self.tokens += r.usage.input_tokens + r.usage.output_tokens
                        in_tok, out_tok = r.usage.input_tokens, r.usage.output_tokens
                self.api_calls += 1
                self.log.append({"temp": temperature, "cached": False,
                                 "latency": round(time.time() - t0, 3),
                                 "in_tok": in_tok, "out_tok": out_tok,
                                 "retries": attempt, "parse": None})
                if use_cache:
                    with _CACHE_LOCK:
                        self.cache[key] = out
                    self._miss_since_save += 1
                    if self._miss_since_save >= 50:  # throttle disk writes
                        _save_cache(self.cfg.cache_path)
                        self._miss_since_save = 0
                return out
            except Exception as e:
                last = e
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"judge call failed after retries: {last}")

    def _parse_and_log(self, raw):
        v, status = _parse_verdict_status(raw)
        if self.log:
            self.log[-1]["parse"] = status
        return v

    def flush_cache(self):
        if self._miss_since_save:
            _save_cache(self.cfg.cache_path)
            self._miss_since_save = 0

    def pairwise(self, x, a, b, criterion, temperature=None):
        user = PAIRWISE_USER.format(x=x, a=a, b=b, criterion=criterion,
                                    criterion_desc=CRITERIA.get(criterion, criterion))
        return self._parse_and_log(self._chat(PAIRWISE_SYS, user,
                              self.cfg.temperature if temperature is None else temperature))

    def cot_pairwise(self, x, a, b, temperature=None):
        """G-Eval-style holistic CoT judge (baseline)."""
        user = COT_USER.format(x=x, a=a, b=b)
        return self._parse_and_log(self._chat(COT_SYS, user,
                              self.cfg.temperature if temperature is None else temperature,
                              max_tokens=256))

    def cot_criterion(self, x, a, b, criterion, temperature=None):
        """Criterion-focused CoT judge (for MPCJ-CoT)."""
        user = PAIRWISE_USER.format(x=x, a=a, b=b, criterion=criterion,
                                    criterion_desc=CRITERIA.get(criterion, criterion))
        return self._parse_and_log(self._chat(COT_CRIT_SYS, user,
                              self.cfg.temperature if temperature is None else temperature,
                              max_tokens=256))


class MockJudge(Judge):
    def __init__(self):
        self.cfg = JudgeConfig(cache_path=None)
        self.calls = 0; self.api_calls = 0; self.tokens = 0; self.cache = {}
        self.log = []; self._miss_since_save = 0
    def _chat(self, system, user, temperature, max_tokens=8):
        self.calls += 1; self.tokens += 50
        self.log.append({"temp": temperature, "cached": False, "latency": 0.0,
                         "in_tok": 40, "out_tok": 10, "retries": 0, "parse": None})
        return "Verdict: A"
    def pairwise(self, x, a, b, criterion, temperature=None):
        self.calls += 1; self.tokens += 50
        if criterion == "conciseness":
            return "A" if len(a) <= len(b) else "B"
        if abs(len(a) - len(b)) < 5:
            return "A"
        return "A" if len(a) > len(b) else "B"
    def cot_pairwise(self, x, a, b, temperature=None):
        self.calls += 1; self.tokens += 50
        return "A" if len(a) >= len(b) else "B"
    def cot_criterion(self, x, a, b, criterion, temperature=None):
        return self.pairwise(x, a, b, criterion, temperature)
