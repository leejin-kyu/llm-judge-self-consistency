# -*- coding: utf-8 -*-
"""P5-34: Unit tests guarding against the cache-collapse pitfall (manuscript Section 6.2).

Run:  python -m pytest tests/ -q   (or: python tests/test_cache_bypass.py)

The stub client returns a fresh output per billed call, so if the cache were
(incorrectly) consulted for stochastic calls, outputs would repeat and the billed
call count would drop below K.
"""
import os, sys, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from judges import Judge, JudgeConfig


class _StubClient:
    """Mimics the OpenAI client shape; every create() is one billed call."""
    def __init__(self):
        self.counter = itertools.count()
        self.chat = self
        self.completions = self

    def create(self, **kw):
        n = next(self.counter)
        class R: pass
        r = R(); r.usage = None
        msg = R(); msg.content = f"Verdict: {'A' if n % 2 == 0 else 'B'} (sample {n})"
        ch = R(); ch.message = msg
        r.choices = [ch]
        return r


def _make_judge(cache_stochastic=False):
    j = Judge(JudgeConfig(model="stub", cache_path=None, cache_stochastic=cache_stochastic))
    j._client = _StubClient()  # bypass _ensure(); no API key needed
    j.cache = {}
    return j


def test_stochastic_calls_bypass_cache():
    """K=5 committee at temperature>0 must bill 5 calls and may produce diverse outputs."""
    j = _make_judge()
    outputs = [j._chat("sys", "same prompt", temperature=0.7) for _ in range(5)]
    assert j.api_calls == 5, f"expected 5 billed calls, got {j.api_calls}"
    assert len(set(outputs)) > 1, "stochastic outputs collapsed to one value"


def test_deterministic_calls_hit_cache():
    """temperature==0 repeats must bill exactly once."""
    j = _make_judge()
    outputs = [j._chat("sys", "same prompt", temperature=0.0) for _ in range(5)]
    assert j.api_calls == 1, f"expected 1 billed call, got {j.api_calls}"
    assert len(set(outputs)) == 1


def test_cache_stochastic_flag_reproduces_pitfall():
    """cache_stochastic=True intentionally reproduces the collapse (Table 7 condition)."""
    j = _make_judge(cache_stochastic=True)
    outputs = [j._chat("sys", "same prompt", temperature=0.7) for _ in range(5)]
    assert j.api_calls == 1, "pitfall condition should bill once"
    assert len(set(outputs)) == 1, "pitfall condition should return byte-identical outputs"


if __name__ == "__main__":
    for fn in (test_stochastic_calls_bypass_cache,
               test_deterministic_calls_hit_cache,
               test_cache_stochastic_flag_reproduces_pitfall):
        fn(); print(f"PASS {fn.__name__}")
