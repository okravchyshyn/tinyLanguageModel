"""Step 9: H x Wout -> softmax -> next-token probabilities."""

from __future__ import annotations

import numpy as np

from src.attention.attention import causal_mask, softmax


def train_output_layer(
    emb: np.ndarray,
    bigrams: np.ndarray,
    steps: int = 400,
    lr: float = 0.05,
    seed: int = 0,
    label_smoothing: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit Wout and bias so softmax(E[i] @ Wout + b) matches the corpus next-word counts.

    A small amount of label smoothing keeps this tiny model from becoming too sharp on sparse
    bigram counts, which usually makes demo predictions a bit less brittle.
    """
    vocab, dim = emb.shape
    counts = bigrams.astype(float).copy()
    counts[0, :] = 0
    counts[:, 0] = 0
    total, rows = counts.sum(), counts.sum(axis=1)
    real_vocab = max(vocab - 1, 1)
    smooth = np.zeros_like(counts)
    smooth[:, 1:] = rows[:, None] / real_vocab
    counts = (1.0 - label_smoothing) * counts + label_smoothing * smooth
    rng = np.random.default_rng(seed)
    w, b = rng.normal(0, 0.01, (dim, vocab)), np.zeros(vocab)
    m = [np.zeros_like(w), np.zeros_like(b)]
    v = [np.zeros_like(w), np.zeros_like(b)]
    loss = 0.0
    for t in range(1, steps + 1):
        logits = emb @ w + b
        logits[:, 0] = -1e9  # never predict <unk>
        p = softmax(logits, axis=1)
        loss = float(-(counts * np.log(p + 1e-12)).sum() / total)
        g = (rows[:, None] * p - counts) / total
        g[:, 0] = 0
        for i, (param, grad) in enumerate(((w, emb.T @ g), (b, g.sum(axis=0)))):
            m[i] = 0.9 * m[i] + 0.1 * grad
            v[i] = 0.999 * v[i] + 0.001 * grad**2
            param -= lr * (m[i] / (1 - 0.9**t)) / (np.sqrt(v[i] / (1 - 0.999**t)) + 1e-8)
    return w, b, loss


def attention_lm_loss_and_grads(emb, ctx, tgt, wq, wk, wv, wout, bias, label_smoothing: float = 0.05):
    """Causal single-head attention LM on windows `ctx` (B, L) -> next ids `tgt` (B, L).

    Hand-written backward pass; embeddings stay frozen. Returns (loss, [dWq, dWk, dWv, dWout, db]).
    """
    x = emb[ctx]
    batch, length, d = x.shape
    q, k, v = x @ wq, x @ wk, x @ wv

    mask = causal_mask(length)
    scaled = q @ k.transpose(0, 2, 1) / np.sqrt(d)
    #a = softmax(np.where(causal_mask(length), -np.inf, scaled))
    a = softmax(np.where(mask, -np.inf, scaled))
    h = a @ v
    logits = h @ wout + bias
    logits[..., 0] = -1e9  # never predict <unk>
    p = softmax(logits)

    weight = (tgt != 0).astype(float)  # skip targets that are <unk>
    n = max(weight.sum(), 1.0)
    rows, cols = np.arange(batch)[:, None], np.arange(length)[None, :]
    target = np.zeros_like(p)
    target[rows, cols, tgt] = 1.0
    if label_smoothing:
        target *= 1.0 - label_smoothing
        target[..., 1:] += label_smoothing / max(target.shape[-1] - 1, 1)
        target[..., 0] = 0.0
        denom = target.sum(axis=-1, keepdims=True)
        target = np.divide(target, denom, out=np.zeros_like(target), where=denom > 0)
    loss = float(-(weight[..., None] * target * np.log(p + 1e-12)).sum() / n)

    dlogits = (p - target) * weight[..., None]
    dlogits *= weight[..., None] / n
    dlogits[..., 0] = 0
    dwout = np.einsum("bld,blv->dv", h, dlogits)
    dh = dlogits @ wout.T
    da = dh @ v.transpose(0, 2, 1)
    dv = a.transpose(0, 2, 1) @ dh
    ds = a * (da - (da * a).sum(-1, keepdims=True)) / np.sqrt(d)
    ds[:, mask] = 0.0
    dq, dk = ds @ k, ds.transpose(0, 2, 1) @ q
    grads = [np.einsum("bld,ble->de", x, g) for g in (dq, dk, dv)] + [dwout, dlogits.sum(axis=(0, 1))]
    return loss, grads


def train_attention_model(
    emb, ids, wq, wk, wv, wout, bias, steps=600, batch=256, window=6, lr=0.01, seed=0, label_smoothing: float = 0.05
) -> tuple[list[np.ndarray], float]:
    """Train Wq, Wk, Wv, Wout and bias with Adam on random windows of the corpus."""
    rng = np.random.default_rng(seed)
    params = [p.copy() for p in (wq, wk, wv, wout, bias)]
    m = [np.zeros_like(p) for p in params]
    s = [np.zeros_like(p) for p in params]
    offsets = np.arange(window + 1)
    loss = 0.0
    for t in range(1, steps + 1):
        idx = rng.integers(0, len(ids) - window - 1, batch)[:, None] + offsets
        loss, grads = attention_lm_loss_and_grads(
            emb, ids[idx[:, :-1]], ids[idx[:, 1:]], *params, label_smoothing=label_smoothing
        )
        for i, g in enumerate(grads):
            m[i] = 0.9 * m[i] + 0.1 * g
            s[i] = 0.999 * s[i] + 0.001 * g**2
            params[i] -= lr * (m[i] / (1 - 0.9**t)) / (np.sqrt(s[i] / (1 - 0.999**t)) + 1e-8)
    return params, loss


def next_token_distribution(h_last: np.ndarray, wout: np.ndarray, bias: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    logits = h_last @ wout + bias
    logits[0] = -np.inf  # <unk> is not a real prediction
    return logits, softmax(logits)


def top_k(probs: np.ndarray, logits: np.ndarray, words: list[str], k: int) -> list[dict]:
    order = np.argsort(-probs)[:k]
    return [
        {"token": words[i], "token_id": int(i), "logit": float(logits[i]), "probability": float(probs[i])}
        for i in order
    ]


def greedy_generate(
    tokens: list[str],
    ids: list[int],
    emb: np.ndarray,
    wq: np.ndarray,
    wk: np.ndarray,
    wv: np.ndarray,
    wout: np.ndarray,
    bias: np.ndarray,
    words: list[str],
    steps: int,
) -> list[dict]:
    """Greedy autoregressive decoding for a few next tokens."""
    from src.attention import attention as att

    generated: list[dict] = []
    work_tokens = list(tokens)
    work_ids = list(ids)
    for _ in range(steps):
        x = emb[work_ids]
        q, k, v = att.compute_qkv(x, wq, wk, wv)
        _, scaled = att.attention_scores(q, k)
        a, _ = att.attention_weights(scaled, causal=True)
        _, h = att.weighted_values(a, v)
        logits, probs = next_token_distribution(h[-1], wout, bias)
        next_id = int(np.argmax(probs))
        next_token = words[next_id]
        generated.append(
            {
                "token": next_token,
                "token_id": next_id,
                "probability": float(probs[next_id]),
                "logit": float(logits[next_id]),
                "context": list(work_tokens),
            }
        )
        work_ids.append(next_id)
        work_tokens.append(next_token)
    return generated
