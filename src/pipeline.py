"""Runs the forward pass, stores every intermediate result, and builds the three views
(math, human, visual) for each stage."""

from __future__ import annotations

import uuid

import numpy as np
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from src import database as db
from src.attention import attention as att
from src.attention.explanations import qkv_human_explanation, token_explanation
from src.embeddings.embedder import cosine, most_similar
from src.model import ModelParams
from src.prediction.predictor import greedy_generate, next_token_distribution, top_k
from src.trace import Trace

STAGES = ["tokenize", "embeddings", "qkv", "scores", "matrix", "weighted", "hidden", "predict"]


def _r(a, nd: int = 4):
    return np.round(a, nd).tolist()


def _dot_string(a: np.ndarray, b: np.ndarray, terms: int = 3) -> str:
    head = " + ".join(f"({a[i]:.2f}×{b[i]:.2f})" for i in range(min(terms, len(a))))
    return f"{head}{' + …' if len(a) > terms else ''} = {float(a @ b):.3f}"


class EmptyTextError(ValueError):
    pass


class Pipeline:
    def __init__(self, params: ModelParams, engine: Engine):
        self.p = params
        self.engine = engine
        self.tokenizer = params.tokenizer

    # ---- computation -------------------------------------------------
    def trace(self, text: str, causal: bool = True, top_n: int = 5) -> Trace:
        tokens, ids = self.tokenizer.encode(text)
        if not tokens:
            raise EmptyTextError("text contains no words")
        p = self.p
        x = p.emb[ids]
        q, k, v = att.compute_qkv(x, p.wq, p.wk, p.wv)
        raw, scaled = att.attention_scores(q, k)
        attn, mask = att.attention_weights(scaled, causal)
        contributions, h = att.weighted_values(attn, v)
        logits, probs = next_token_distribution(h[-1], p.wout, p.bias)
        return Trace(
            run_id=uuid.uuid4().hex, text=text, causal=causal, tokens=tokens, ids=ids,
            x=x, q=q, k=k, v=v, raw=raw, scaled=scaled, mask=mask, attn=attn,
            contributions=contributions, h=h, logits=logits, probs=probs,
            top=top_k(probs, logits, p.words, top_n),
        )

    def persist(self, tr: Trace, upto: str) -> None:
        n = STAGES.index(upto)
        base = {"run_id": tr.run_id, "text": tr.text}
        rows: list = []

        def per_token(model, mat):
            return [model(position=i, word=w, vector=_r(mat[i], 6), **base) for i, w in enumerate(tr.tokens)]

        if n >= 2:
            rows += per_token(db.Query, tr.q) + per_token(db.Key, tr.k) + per_token(db.Value, tr.v)
        if n >= 3:
            rows.append(db.AttentionScore(raw=_r(tr.raw, 6), scaled=_r(tr.scaled, 6), **base))
        if n >= 4:
            rows.append(db.AttentionMatrix(matrix=_r(tr.attn, 6), causal=tr.causal, **base))
        if n >= 5:
            rows += per_token(db.WeightedValue, tr.contributions)
        if n >= 6:
            rows += per_token(db.HiddenState, tr.h)
        if n >= 7:
            rows += [
                db.Prediction(rank=i + 1, token_id=t["token_id"], token=t["token"], logit=t["logit"],
                              probability=t["probability"], **base)
                for i, t in enumerate(tr.top)
            ]
        with Session(self.engine) as s:
            s.add_all(rows)
            s.commit()

    def run(self, text: str, upto: str = "predict", causal: bool = True, top_n: int = 5) -> Trace:
        tr = self.trace(text, causal, top_n)
        self.persist(tr, upto)
        return tr

    # ---- stage views -------------------------------------------------
    def tokenization(self, tr: Trace) -> dict:
        unknown = [t for t, i in zip(tr.tokens, tr.ids) if i == 0]
        coverage = 1.0 - (len(unknown) / max(len(tr.ids), 1))
        return {
            "run_id": tr.run_id, "text": tr.text, "tokens": tr.tokens, "ids": tr.ids, "unknown_tokens": unknown,
            "coverage": {
                "known_token_count": len(tr.ids) - len(unknown),
                "unknown_token_count": len(unknown),
                "vocabulary_coverage": round(coverage, 4),
            },
            "math_view": {"lookup": [{"token": t, "id": i} for t, i in zip(tr.tokens, tr.ids)]},
            "human_view": (
                "The model cannot read letters, only numbers. Each word is looked up in the vocabulary and replaced "
                "by its id. Words outside the vocabulary become <unk> (id 0), so the model knows nothing about them."
            ),
            "visual_view": None,
        }

    def embeddings(self, tr: Trace, similar_n: int = 5) -> dict:
        details = []
        for w, i in zip(tr.tokens, tr.ids):
            vec = self.p.emb[i]
            sims = most_similar(self.p.emb, self.p.words, i, similar_n) if i else []
            details.append({
                "word": w, "token_id": i, "vector": _r(vec), "vector_length": round(float(np.linalg.norm(vec)), 4),
                "similar_words": [{"word": sw, "cosine": round(c, 4)} for sw, c in sims],
            })
        pairs = [
            {"a": tr.tokens[a], "b": tr.tokens[b], "cosine": round(cosine(tr.x[a], tr.x[b]), 4)}
            for a in range(len(tr.tokens)) for b in range(a + 1, len(tr.tokens))
        ]
        return {
            "run_id": tr.run_id,
            "embeddings": {w: _r(self.p.emb[i]) for w, i in zip(tr.tokens, tr.ids)},
            "dimension": self.p.dim, "details": details,
            "math_view": {
                "formula": "vector = E[token_id]  (row of an embedding matrix of shape vocab x d)",
                "vector_length": "||v|| = sqrt(v1^2 + ... + vd^2)",
                "cosine_similarity": "cos(a, b) = a.b / (||a|| ||b||)",
                "pairwise_cosine": pairs,
            },
            "human_view": (
                "An id is just a label. An embedding gives each word a short list of numbers so that words used in "
                "similar contexts (e.g. 'cat' and 'dog') end up close together. Cosine similarity measures how closely "
                "two vectors point in the same direction."
            ),
            "visual_view": {"png": "/visualize/embeddings"},
        }

    def qkv(self, tr: Trace) -> dict:
        details = [token_explanation(w, tr.q[i], tr.k[i], tr.v[i]) for i, w in enumerate(tr.tokens)]
        d = self.p.dim
        return {
            "run_id": tr.run_id, "tokens": tr.tokens,
            "q": {"matrix": _r(tr.q)}, "k": {"matrix": _r(tr.k)}, "v": {"matrix": _r(tr.v)},
            "human_explanation": qkv_human_explanation(tr.tokens, details),
            "per_token": details,
            "math_view": {
                "formulas": "Q = X.Wq, K = X.Wk, V = X.Wv",
                "shapes": f"X: {tr.x.shape}, Wq/Wk/Wv: ({d}, {d}), Q/K/V: {tr.q.shape}",
                "Wq": _r(self.p.wq), "Wk": _r(self.p.wk), "Wv": _r(self.p.wv),
                "worked_example": {
                    "token": tr.tokens[-1],
                    "q[0]": "x . Wq[:,0] = " + _dot_string(tr.x[-1], self.p.wq[:, 0]),
                    "k[0]": "x . Wk[:,0] = " + _dot_string(tr.x[-1], self.p.wk[:, 0]),
                    "v[0]": "x . Wv[:,0] = " + _dot_string(tr.x[-1], self.p.wv[:, 0]),
                },
            },
            "visual_view": None,
        }

    def scores(self, tr: Trace) -> dict:
        n, d = len(tr.tokens), self.p.dim
        dots = [
            {"query": tr.tokens[i], "key": tr.tokens[j], "calculation": _dot_string(tr.q[i], tr.k[j])
             + f"; / sqrt({d}) = {tr.scaled[i, j]:.3f}"}
            for i in range(n) for j in range(n)
        ]
        return {
            "run_id": tr.run_id, "tokens": tr.tokens, "raw_scores": _r(tr.raw), "scaled_scores": _r(tr.scaled),
            "math_view": {"formula": f"scores = Q.K^T / sqrt(d_k), d_k = {d}", "dot_products": dots},
            "human_view": (
                "Each token's Query is compared with every token's Key using a dot product. A large number means "
                "'what I'm looking for matches what you offer'. Dividing by sqrt(d_k) keeps numbers small so softmax "
                "does not become too extreme."
            ),
            "visual_view": {"png": "/visualize/attention-scores"},
        }

    def matrix(self, tr: Trace) -> dict:
        n = len(tr.tokens)
        calc = []
        for i in range(n):
            peak = tr.scaled[i][~tr.mask[i]].max()
            exps = np.where(tr.mask[i], 0.0, np.exp(tr.scaled[i] - peak))
            calc.append({
                "token": tr.tokens[i],
                "scaled_scores": [None if m else round(float(s), 4) for s, m in zip(tr.scaled[i], tr.mask[i])],
                "exp(score - max)": _r(exps), "sum": round(float(exps.sum()), 4), "probabilities": _r(tr.attn[i]),
            })
        return {
            "run_id": tr.run_id, "tokens": tr.tokens, "causal_mask": tr.causal, "attention": _r(tr.attn),
            "distribution": [
                {"token": tr.tokens[i], "attends_to": {tr.tokens[j]: round(float(tr.attn[i, j]), 4) for j in range(n)}}
                for i in range(n)
            ],
            "math_view": {"formula": "A[i,j] = exp(s[i,j]) / sum_j exp(s[i,j])", "rows": calc},
            "human_view": (
                "Softmax turns each row of raw scores into probabilities that add up to 1: how much attention a token "
                "pays to each other token. "
                + ("A causal mask hides later tokens, because when predicting the next word the model may not peek ahead. "
                   if tr.causal else "")
            ),
            "visual_view": {"png": "/visualize/attention-matrix"},
        }

    def weighted(self, tr: Trace) -> dict:
        n = len(tr.tokens)
        return {
            "run_id": tr.run_id, "tokens": tr.tokens,
            "weighted_values": [
                {"token": tr.tokens[i], "contributions": {tr.tokens[j]: _r(tr.contributions[i, j]) for j in range(n)}}
                for i in range(n)
            ],
            "math_view": {"formula": "contribution[i,j] = A[i,j] * V[j]; summing over j gives Attention.V"},
            "human_view": (
                "Every token takes a share of every visible token's Value, in proportion to its attention weight. "
                "Important tokens contribute a lot, ignored tokens almost nothing."
            ),
            "visual_view": None,
        }

    def hidden_state(self, tr: Trace) -> dict:
        diff = tr.h - tr.x
        return {
            "run_id": tr.run_id, "tokens": tr.tokens,
            "previous_embedding": _r(tr.x), "hidden_state": _r(tr.h), "difference": _r(diff),
            "change_per_token": [
                {"token": w, "difference_length": round(float(np.linalg.norm(diff[i])), 4),
                 "cosine_to_previous": round(cosine(tr.x[i], tr.h[i]), 4)}
                for i, w in enumerate(tr.tokens)
            ],
            "math_view": {"formula": "H = Attention . V;  difference = H - X", "shape": list(tr.h.shape)},
            "human_view": (
                "The embedding of a word is the same in every sentence. H is the same word after it has gathered "
                "information from its context, so it now depends on the surrounding words. The difference shows how "
                "much context changed each token."
            ),
            "visual_view": {"png": "/visualize/hidden-state"},
        }

    def prediction(self, tr: Trace) -> dict:
        shown = tr.top
        unk_count = sum(1 for token_id in tr.ids if token_id == 0)
        coverage = 1.0 - (unk_count / max(len(tr.ids), 1))
        warning = None
        if unk_count:
            warning = (
                f"{unk_count} of {len(tr.ids)} input tokens are out of vocabulary and were mapped to <unk>. "
                "Predictions may look unreasonable because this tiny word-level model can only use words it has seen often enough in the training corpus."
            )
        return {
            "run_id": tr.run_id, "text": tr.text,
            "top_predictions": [{"token": t["token"], "probability": round(t["probability"], 4)} for t in shown],
            "input_diagnostics": {
                "tokens": tr.tokens,
                "token_ids": tr.ids,
                "vocabulary_coverage": round(coverage, 4),
                "unknown_token_count": unk_count,
                "warning": warning,
            },
            "math_view": {
                "formula": "logits = H[last] . Wout + b;  probabilities = softmax(logits)",
                "last_token": tr.tokens[-1],
                "logits": {t["token"]: round(t["logit"], 4) for t in shown},
                "probability_sum_check": round(float(tr.probs.sum()), 6),
            },
            "human_view": (
                f'Only the last token ("{tr.tokens[-1]}") is used: its context-aware vector is scored against every '
                "vocabulary word, and softmax converts the scores into probabilities. The tiny model has only seen "
                "word-pair statistics from the corpus, so suggestions reflect common word pairs, not real understanding. "
                + ("Some of your input words are outside the vocabulary and were replaced with <unk>, which usually makes the result much worse."
                   if unk_count else "")
            ),
            "visual_view": {"png": "/visualize/predictions"},
        }

    def generate(self, text: str, steps: int = 3) -> dict:
        tr = self.trace(text, causal=True, top_n=5)
        generated = greedy_generate(
            tr.tokens,
            tr.ids,
            self.p.emb,
            self.p.wq,
            self.p.wk,
            self.p.wv,
            self.p.wout,
            self.p.bias,
            self.p.words,
            steps,
        )
        return {
            "run_id": tr.run_id,
            "text": text,
            "steps": steps,
            "generated": generated,
            "final_text": " ".join(tr.tokens + [g["token"] for g in generated]),
        }

    def explain(self, tr: Trace, similar_n: int = 5) -> dict:
        return {
            "run_id": tr.run_id,
            "tokenization": self.tokenization(tr),
            "embeddings": self.embeddings(tr, similar_n),
            "qkv": self.qkv(tr),
            "attention": {"scores": self.scores(tr), "matrix": self.matrix(tr), "weighted_values": self.weighted(tr)},
            "hidden_state": self.hidden_state(tr),
            "predictions": self.prediction(tr),
        }
