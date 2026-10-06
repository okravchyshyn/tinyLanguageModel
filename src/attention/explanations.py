"""Plain-English explanations for Q/K/V."""

from __future__ import annotations

import numpy as np

QUERY_MEANING = "What information am I looking for?"
KEY_MEANING = "What information can I provide?"
VALUE_MEANING = "What information should be shared if I am selected as important?"

# word -> (query hint, key hint, value hint); hand-written for the demo words.
ROLE_HINTS = {
    "cat": ("looking for an action or description of what it does", "an animal / subject", "animal and subject information"),
    "dog": ("looking for an action or description of what it does", "an animal / subject", "animal and subject information"),
    "run": ("looking for the subject performing the action", "describes an action", "movement / action information"),
    "eat": ("looking for who eats and what is eaten", "describes an action", "eating / action information"),
    "fast": ("looking for the action being described", "describes speed or manner", "speed / manner information"),
    "food": ("looking for the action that involves it", "an object that can be eaten", "food / object information"),
}
GENERIC_HINT = ("looking for context that clarifies my role", "a word with its own meaning", "the meaning of this word")


def token_explanation(word: str, q: np.ndarray, k: np.ndarray, v: np.ndarray) -> dict:
    qh, kh, vh = ROLE_HINTS.get(word, GENERIC_HINT)
    return {
        "token": word,
        "query": {"question": QUERY_MEANING, "meaning": qh, "vector_length": round(float(np.linalg.norm(q)), 4)},
        "key": {"question": KEY_MEANING, "meaning": kh, "vector_length": round(float(np.linalg.norm(k)), 4)},
        "value": {"question": VALUE_MEANING, "meaning": vh, "vector_length": round(float(np.linalg.norm(v)), 4)},
    }


def qkv_human_explanation(tokens: list[str], details: list[dict]) -> str:
    lines = [
        f'Sentence: "{" ".join(tokens)}". Every token is turned into three vectors:',
        f"- Query: {QUERY_MEANING}",
        f"- Key: {KEY_MEANING}",
        f"- Value: {VALUE_MEANING}",
        "(The role descriptions below are hand-written hints; the vectors come from matrices trained on the corpus.)",
    ]
    for d in details:
        lines.append(
            f'Token "{d["token"]}": Query = {d["query"]["meaning"]}; '
            f'Key = {d["key"]["meaning"]}; Value = {d["value"]["meaning"]}.'
        )
    return "\n".join(lines)
