"""Visual view: heatmaps and bar charts rendered to PNG bytes."""

from __future__ import annotations

import io

import numpy as np
from matplotlib.figure import Figure

from src.trace import Trace

STAGES = ["embeddings", "attention-scores", "attention-matrix", "hidden-state", "predictions"]


def _png(fig: Figure) -> bytes:
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    return buf.getvalue()


def _heat(ax, fig, m, xlabels, ylabels, title, annotate=True, cmap="viridis"):
    im = ax.imshow(m, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(xlabels)), xlabels, rotation=45, ha="right")
    ax.set_yticks(range(len(ylabels)), ylabels)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    if annotate:
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", color="white", fontsize=8)


def heatmap(m: np.ndarray, xlabels, ylabels, title: str, cmap="viridis") -> bytes:
    fig = Figure(figsize=(1.2 + 0.9 * m.shape[1], 1.5 + 0.7 * m.shape[0]))
    _heat(fig.subplots(), fig, m, xlabels, ylabels, title, annotate=m.size <= 100, cmap=cmap)
    return _png(fig)


def bar_chart(labels, values, title: str, ylabel: str) -> bytes:
    fig = Figure(figsize=(6, 3.5))
    ax = fig.subplots()
    ax.bar(labels, values, color="steelblue")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    return _png(fig)


def render(stage: str, tr: Trace, words: list[str], emb: np.ndarray) -> bytes:
    if stage == "embeddings":
        dims = [f"d{i}" for i in range(emb.shape[1])]
        return heatmap(tr.x, dims, tr.tokens, "Embedding vectors (one row per word)", cmap="coolwarm")
    if stage == "attention-scores":
        return heatmap(tr.scaled, tr.tokens, tr.tokens, "Scaled attention scores (query rows x key columns)")
    if stage == "attention-matrix":
        return heatmap(tr.attn, tr.tokens, tr.tokens, "Attention weights (softmax; rows sum to 1)", cmap="Blues")
    if stage == "hidden-state":
        fig = Figure(figsize=(13, 1.8 + 0.6 * len(tr.tokens)))
        axes = fig.subplots(1, 3)
        dims = [f"d{i}" for i in range(tr.x.shape[1])]
        for ax, mat, title in zip(axes, (tr.x, tr.h, tr.h - tr.x), ("Previous embedding X", "Context-aware H", "Difference H - X")):
            _heat(ax, fig, mat, dims, tr.tokens, title, annotate=False, cmap="coolwarm")
        return _png(fig)
    if stage == "predictions":
        return bar_chart([t["token"] for t in tr.top], [t["probability"] for t in tr.top],
                         f'Next-token probabilities after "{tr.text}"', "probability")
    raise KeyError(stage)
