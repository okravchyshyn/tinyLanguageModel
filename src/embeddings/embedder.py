"""Step 3: small educational embeddings.

Words that appear near the same neighbours get similar vectors:
co-occurrence counts -> PPMI -> truncated SVD (the classic count-based recipe).
"""

from __future__ import annotations

import numpy as np
from sqlalchemy import delete
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from src.database import Embedding

TARGET_NORM = 2.0


def cooccurrence_matrix(ids: np.ndarray, vocab_size: int, window: int = 2) -> np.ndarray:
    counts = np.zeros((vocab_size, vocab_size))
    for k in range(1, window + 1):
        flat = np.bincount(ids[:-k] * vocab_size + ids[k:], minlength=vocab_size * vocab_size)
        counts += flat.reshape(vocab_size, vocab_size)
    counts += counts.T
    counts[0, :] = 0  # <unk> carries no meaning
    counts[:, 0] = 0
    return counts


def ppmi_svd_embeddings(counts: np.ndarray, dim: int) -> np.ndarray:
    total = counts.sum()
    row = counts.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log(counts * total / np.outer(row, row))
    ppmi = np.nan_to_num(np.maximum(pmi, 0.0), nan=0.0, posinf=0.0, neginf=0.0)
    u, s, _ = np.linalg.svd(ppmi)
    emb = u[:, :dim] * np.sqrt(s[:dim])
    emb[0] = 0.0
    mean_norm = np.linalg.norm(emb[1:], axis=1).mean()
    return emb * (TARGET_NORM / mean_norm)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denom) if denom > 0 else 0.0


def most_similar(emb: np.ndarray, words: list[str], token_id: int, k: int) -> list[tuple[str, float]]:
    norms = np.linalg.norm(emb, axis=1)
    norms[norms == 0] = 1.0
    sims = (emb / norms[:, None]) @ (emb[token_id] / norms[token_id])
    sims[0] = -np.inf  # skip <unk>
    sims[token_id] = -np.inf
    order = np.argsort(-sims)[:k]
    return [(words[i], float(sims[i])) for i in order]


def save_embeddings(engine: Engine, emb: np.ndarray) -> None:
    with Session(engine) as s:
        s.execute(delete(Embedding))
        s.add_all(Embedding(token_id=i, vector=np.round(v, 6).tolist()) for i, v in enumerate(emb))
        s.commit()
