# -*- coding: utf-8 -*-
"""Data loading. Includes a tiny built-in SAMPLE for smoke tests, plus a loader
stub for real MT-Bench / LLMBar pairwise human-preference data.

각 항목: {"x": instruction, "a": resp_A, "b": resp_B, "gold": "A"|"B"|"tie"}
gold = 인간 선호 라벨.
"""
from __future__ import annotations
import json, os

# --- tiny illustrative sample (파이프라인 점검용; 실제 결과 아님) ---
SAMPLE = [
    {"x": "Explain why the sky is blue in one sentence.",
     "a": "Sunlight scatters off air molecules, and blue light scatters more than red because it has a shorter wavelength (Rayleigh scattering).",
     "b": "Because of the air.", "gold": "A"},
    {"x": "Give a Python one-liner to reverse a string.",
     "a": "s[::-1]",
     "b": "You can reverse a string in Python by using slicing with a step of -1, which is a very common and idiomatic approach widely taught in tutorials: s[::-1].",
     "gold": "A"},  # A is correct & concise; tests verbosity bias (B is padded)
    {"x": "What is the capital of Australia?",
     "a": "Sydney.", "b": "Canberra.", "gold": "B"},
    {"x": "Summarize: cats are independent pets that still seek affection.",
     "a": "Cats are independent yet affectionate pets.",
     "b": "Cats are pets.", "gold": "A"},
    {"x": "Is 17 a prime number? Answer yes/no with reason.",
     "a": "Yes; 17 has no divisors other than 1 and 17.",
     "b": "No, 17 is divisible by 3.", "gold": "A"},
]


def load_sample():
    return list(SAMPLE)


def load_jsonl(path):
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def load_mtbench(path=None):
    """실데이터 사용 시:
    MT-Bench human judgments(예: mt_bench_human_judgments)에서
    pairwise 항목을 위 스키마(jsonl)로 변환해 load_jsonl 로 읽으세요.
    참고: Zheng et al. 2023, https://github.com/lm-sys/FastChat
    """
    if path and os.path.exists(path):
        return load_jsonl(path)
    raise FileNotFoundError(
        "MT-Bench pairwise jsonl이 없습니다. data.py 주석 참고하여 변환 후 경로를 지정하세요.")
