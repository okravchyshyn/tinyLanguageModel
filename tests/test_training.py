import numpy as np

from src.attention.attention import init_projections
from src.prediction.predictor import attention_lm_loss_and_grads, train_attention_model


def _setup():
    rng = np.random.default_rng(0)
    vocab, d = 12, 4
    emb = rng.normal(size=(vocab, d))
    emb[0] = 0
    wq, wk, wv = init_projections(d, 1)
    wout, bias = rng.normal(size=(d, vocab)), rng.normal(size=vocab)
    ids = rng.integers(1, vocab, 400)
    return emb, ids, [wq, wk, wv, wout, bias]


def test_gradients_match_finite_differences():
    emb, ids, params = _setup()
    ctx, tgt = ids[:12].reshape(2, 6), ids[1:13].reshape(2, 6)
    _, grads = attention_lm_loss_and_grads(emb, ctx, tgt, *params)
    eps = 1e-6
    for p, g in zip(params, grads):
        i = tuple(0 for _ in p.shape)
        p[i] += eps
        hi, _ = attention_lm_loss_and_grads(emb, ctx, tgt, *params)
        p[i] -= 2 * eps
        lo, _ = attention_lm_loss_and_grads(emb, ctx, tgt, *params)
        p[i] += eps
        assert abs((hi - lo) / (2 * eps) - g[i]) < 1e-5


def test_training_reduces_loss():
    emb, ids, params = _setup()
    ids = np.tile(np.arange(1, 6), 80)  # fully predictable sequence
    before, _ = attention_lm_loss_and_grads(emb, ids[:60].reshape(10, 6), ids[1:61].reshape(10, 6), *params)
    _, after = train_attention_model(emb, ids, *params, steps=150, batch=32)
    assert after < before
