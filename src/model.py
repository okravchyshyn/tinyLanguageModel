"""The set of weights that make up the tiny model, plus build/save/load."""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sqlalchemy.engine import Engine

from src import config
from src.attention.attention import init_projections
from src.embeddings.embedder import cooccurrence_matrix, ppmi_svd_embeddings, save_embeddings
from src.prediction.predictor import train_attention_model, train_output_layer
from src.tokenization.vocabulary import (
    Tokenizer,
    build_vocabulary,
    read_corpus_words,
    save_vocabulary,
    words_to_ids,
)

log = logging.getLogger(__name__)


@dataclass
class ModelParams:
    words: list[str]  # index == token id
    emb: np.ndarray  # (V, d)
    wq: np.ndarray
    wk: np.ndarray
    wv: np.ndarray
    wout: np.ndarray  # (d, V)
    bias: np.ndarray  # (V,)

    @property
    def dim(self) -> int:
        return self.emb.shape[1]

    @property
    def tokenizer(self) -> Tokenizer:
        return Tokenizer(self.words)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, words=np.array(self.words), emb=self.emb, wq=self.wq, wk=self.wk,
                 wv=self.wv, wout=self.wout, bias=self.bias)

    @classmethod
    def load(cls, path: Path) -> "ModelParams":
        with np.load(path) as f:
            return cls([str(w) for w in f["words"]], f["emb"], f["wq"], f["wk"], f["wv"], f["wout"], f["bias"])


def build_model(
    engine: Engine,
    corpus_path: Path = config.CORPUS_PATH,
    vocab_size: int = config.VOCAB_SIZE,
    dim: int = config.EMBED_DIM,
    seed: int = config.SEED,
    model_path: Path = config.MODEL_PATH,
    train_steps: int = config.TRAIN_STEPS,
    train_window: int = config.TRAIN_WINDOW,
) -> ModelParams:
    """Corpus -> vocabulary (DB) -> embeddings (DB) -> output layer; weights saved to data/processed."""
    if not corpus_path.exists() and corpus_path == config.CORPUS_PATH:
        corpus_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(config.SOURCE_CORPUS, corpus_path)

    words = read_corpus_words(corpus_path)
    entries = build_vocabulary(words, vocab_size)
    save_vocabulary(engine, entries)
    id_words = [e.word for e in entries]

    ids = words_to_ids(words, {w: i for i, w in enumerate(id_words)})
    size = len(entries)
    emb = ppmi_svd_embeddings(cooccurrence_matrix(ids, size), dim)
    save_embeddings(engine, emb)

    bigrams = np.bincount(ids[:-1] * size + ids[1:], minlength=size * size).reshape(size, size)
    wout, bias, loss = train_output_layer(emb, bigrams)
    log.info("output layer (bigram warm start) cross-entropy %.3f", loss)

    wq, wk, wv = init_projections(dim, seed)
    (wq, wk, wv, wout, bias), loss = train_attention_model(
        emb, ids, wq, wk, wv, wout, bias, steps=train_steps, window=train_window, seed=seed
    )
    log.info("Wq/Wk/Wv/Wout trained, cross-entropy %.3f", loss)
    params = ModelParams(id_words, emb, wq, wk, wv, wout, bias)
    params.save(model_path)
    return params
