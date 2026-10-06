"""Step 1 and 2: build the word-level vocabulary and turn text into token ids."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from sqlalchemy import delete
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from src.config import UNK
from src.database import Vocabulary

WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")
BOOK_HEADER_RE = re.compile(r"^=== .* ===$")


@dataclass(frozen=True)
class VocabEntry:
    token_id: int
    word: str
    frequency: int
    rank: int


def split_words(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def read_corpus_words(path: Path) -> list[str]:
    """Read the merged corpus, skipping the `=== book.txt ===` separator lines."""
    words: list[str] = []
    with open(path, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if BOOK_HEADER_RE.match(line.strip()):
                continue
            words.extend(split_words(line))
    return words


def build_vocabulary(words: Iterable[str], size: int) -> list[VocabEntry]:
    """`<unk>` is id 0 (rank 0); real words get id == rank, most frequent first."""
    counts = Counter(words)
    top = counts.most_common(size)
    kept = {w for w, _ in top}
    unk_freq = sum(c for w, c in counts.items() if w not in kept)
    entries = [VocabEntry(0, UNK, unk_freq, 0)]
    entries += [VocabEntry(i, w, c, i) for i, (w, c) in enumerate(top, start=1)]
    return entries


def save_vocabulary(engine: Engine, entries: list[VocabEntry]) -> None:
    with Session(engine) as s:
        s.execute(delete(Vocabulary))
        s.add_all(Vocabulary(token_id=e.token_id, word=e.word, frequency=e.frequency, rank=e.rank) for e in entries)
        s.commit()


def words_to_ids(words: Iterable[str], word_to_id: dict[str, int]) -> np.ndarray:
    return np.fromiter((word_to_id.get(w, 0) for w in words), dtype=np.int64)


class Tokenizer:
    def __init__(self, id_to_word: list[str]):
        self.id_to_word = id_to_word
        self.word_to_id = {w: i for i, w in enumerate(id_to_word)}

    def encode(self, text: str) -> tuple[list[str], list[int]]:
        tokens = split_words(text)
        return tokens, [self.word_to_id.get(t, 0) for t in tokens]

    def decode(self, ids: Iterable[int]) -> list[str]:
        return [self.id_to_word[i] for i in ids]


def load_vocabulary(engine: Engine, limit: Optional[int] = None) -> list[VocabEntry]:
    with Session(engine) as s:
        q = s.query(Vocabulary).order_by(Vocabulary.rank)
        if limit:
            q = q.limit(limit)
        return [VocabEntry(v.token_id, v.word, v.frequency, v.rank) for v in q]
