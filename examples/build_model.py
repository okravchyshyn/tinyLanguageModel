"""Build the vocabulary, embeddings and output layer from the corpus (stores them in the SQLite DB).

Usage (project root):  python examples/build_model.py
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.database import make_engine  # noqa: E402
from src.model import build_model  # noqa: E402

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    params = build_model(make_engine())
    print(f"Built model: {len(params.words)} tokens, embedding dimension {params.dim}")
