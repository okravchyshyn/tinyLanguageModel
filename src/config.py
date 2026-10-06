"""Paths and hyper-parameters shared by the whole demo."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
CORPUS_PATH = DATA_DIR / "all_books.txt"
SOURCE_CORPUS = ROOT / "PDF_txt_convert" / "output" / "merged_txt" / "all_books.txt"
DB_PATH = ROOT / "database" / "llm_demo.db"
MODEL_PATH = PROCESSED_DIR / "model.npz"

VOCAB_SIZE = 1000  # real words (~500; large enough to include the demo word "fast"); <unk> is added as token 0
EMBED_DIM = 32  # intentionally tiny so vectors fit on screen
SEED = 42
UNK = "<unk>"
TRAIN_STEPS = 1500  # reduced so first-time `python examples/build_model.py` is much faster in local demos
TRAIN_WINDOW = 12  # words per training window
