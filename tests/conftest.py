"""Tiny synthetic corpus + a small model built once per test session."""

import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.database import make_engine
from src.model import build_model

CORPUS = ("the cat run fast . the dog eat food . " * 40 + "a cat eat food and a dog run fast . " * 40)


@pytest.fixture(scope="session")
def built(tmp_path_factory):
    d = tmp_path_factory.mktemp("demo")
    corpus = d / "corpus.txt"
    corpus.write_text("=== book.txt ===\n" + CORPUS)
    engine = make_engine(d / "demo.db")
    params = build_model(engine, corpus_path=corpus, vocab_size=20, dim=8, model_path=d / "model.npz", train_steps=50)
    return d, engine, params


@pytest.fixture(scope="session")
def client(built):
    d, _, _ = built
    with TestClient(create_app(d / "demo.db", d / "model.npz")) as c:
        yield c
