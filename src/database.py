"""SQLite schema: one table per stage so every intermediate result can be inspected."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import JSON, Float, ForeignKey, Integer, String, create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Vocabulary(Base):
    __tablename__ = "vocabulary"
    token_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    word: Mapped[str] = mapped_column(String, unique=True)
    frequency: Mapped[int] = mapped_column(Integer)
    rank: Mapped[int] = mapped_column(Integer)


class Embedding(Base):
    __tablename__ = "embeddings"
    token_id: Mapped[int] = mapped_column(ForeignKey("vocabulary.token_id"), primary_key=True)
    vector: Mapped[Any] = mapped_column(JSON)


class _RunRow:
    """Columns shared by every per-request table."""

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String, index=True)
    text: Mapped[str] = mapped_column(String)
    created_at: Mapped[dt.datetime] = mapped_column(default=dt.datetime.utcnow)


class _TokenRow(_RunRow):
    position: Mapped[int] = mapped_column(Integer)
    word: Mapped[str] = mapped_column(String)
    vector: Mapped[Any] = mapped_column(JSON)


class Query(_TokenRow, Base):
    __tablename__ = "queries"


class Key(_TokenRow, Base):
    __tablename__ = "keys"


class Value(_TokenRow, Base):
    __tablename__ = "values"


class WeightedValue(_TokenRow, Base):
    """`vector` holds the n x d contributions attention[i, j] * V[j] for token i."""

    __tablename__ = "weighted_values"


class HiddenState(_TokenRow, Base):
    __tablename__ = "hidden_states"


class AttentionScore(_RunRow, Base):
    __tablename__ = "attention_scores"
    raw: Mapped[Any] = mapped_column(JSON)
    scaled: Mapped[Any] = mapped_column(JSON)


class AttentionMatrix(_RunRow, Base):
    __tablename__ = "attention_matrices"
    matrix: Mapped[Any] = mapped_column(JSON)
    causal: Mapped[bool] = mapped_column(default=True)


class Prediction(_RunRow, Base):
    __tablename__ = "predictions"
    rank: Mapped[int] = mapped_column(Integer)
    token_id: Mapped[int] = mapped_column(Integer)
    token: Mapped[str] = mapped_column(String)
    logit: Mapped[float] = mapped_column(Float)
    probability: Mapped[float] = mapped_column(Float)


def make_engine(path: Optional[Path] = None) -> Engine:
    if path is None:
        from src.config import DB_PATH

        path = DB_PATH
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{path}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    return engine
