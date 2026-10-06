"""Steps 4-8: Q/K/V, attention scores, softmax, weighted values (single head)."""

from __future__ import annotations

import numpy as np


def init_projections(dim: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Wq, Wk, Wv start random (Wv near identity so H stays comparable to the embeddings)."""
    rng = np.random.default_rng(seed)
    wq = rng.normal(0, 1 / np.sqrt(dim), (dim, dim))
    wk = rng.normal(0, 1 / np.sqrt(dim), (dim, dim))
    wv = np.eye(dim) + rng.normal(0, 0.1 / np.sqrt(dim), (dim, dim))
    return wq, wk, wv


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=axis, keepdims=True)


def compute_qkv(x, wq, wk, wv):
    return x @ wq, x @ wk, x @ wv


def attention_scores(q: np.ndarray, k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (raw Q.K^T, scaled by sqrt(d_k))."""
    raw = q @ k.T
    return raw, raw / np.sqrt(q.shape[-1])


def causal_mask(n: int) -> np.ndarray:
    """True where a token is NOT allowed to look (the future)."""
    return np.triu(np.ones((n, n), dtype=bool), k=1)


def attention_weights(scaled: np.ndarray, causal: bool) -> tuple[np.ndarray, np.ndarray]:
    mask = causal_mask(scaled.shape[0]) if causal else np.zeros(scaled.shape, dtype=bool)
    return softmax(np.where(mask, -np.inf, scaled), axis=-1), mask


def weighted_values(a: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """contributions[i, j] = a[i, j] * v[j]; H[i] = sum_j contributions[i, j] (= A @ V)."""
    contributions = a[:, :, None] * v[None, :, :]
    return contributions, contributions.sum(axis=1)
