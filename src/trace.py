"""Intermediate results of one forward pass, kept as NumPy arrays."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Trace:
    run_id: str
    text: str
    causal: bool
    tokens: list[str]
    ids: list[int]
    x: np.ndarray  # embeddings (n, d)
    q: np.ndarray
    k: np.ndarray
    v: np.ndarray
    raw: np.ndarray  # Q K^T (n, n)
    scaled: np.ndarray  # raw / sqrt(d)
    mask: np.ndarray  # True = hidden from attention
    attn: np.ndarray  # softmax(scaled) (n, n)
    contributions: np.ndarray  # (n, n, d)
    h: np.ndarray  # (n, d)
    logits: np.ndarray  # (V,)
    probs: np.ndarray  # (V,)
    top: list[dict] = field(default_factory=list)
