"""Worked examples: print every stage for "cat run" and "dog eat" and save the charts as PNGs.

Usage (project root):  python examples/run_demo.py ["custom text" ...]
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src import config  # noqa: E402
from src.database import make_engine  # noqa: E402
from src.model import ModelParams, build_model  # noqa: E402
from src.pipeline import Pipeline  # noqa: E402
from src.visualization import plots  # noqa: E402

OUT = ROOT / "examples" / "output"


def show(title: str, payload: dict) -> None:
    print(f"\n=== {title} ===")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def main(texts: list[str]) -> None:
    engine = make_engine()
    params = ModelParams.load(config.MODEL_PATH) if config.MODEL_PATH.exists() else build_model(engine)
    pipe = Pipeline(params, engine)
    OUT.mkdir(exist_ok=True)
    np.set_printoptions(precision=3, suppress=True)
    for text in texts:
        print(f"\n########## {text!r} ##########")
        tr = pipe.run(text)
        report = pipe.explain(tr)
        show("1. Tokenization", report["tokenization"])
        show("2. Embeddings", report["embeddings"])
        show("3. Q/K/V", report["qkv"])
        show("4. Attention scores", report["attention"]["scores"])
        show("5. Attention matrix", report["attention"]["matrix"])
        show("6. Weighted values", report["attention"]["weighted_values"])
        show("7. Hidden state", report["hidden_state"])
        show("8. Prediction", report["predictions"])
        for stage in plots.STAGES:
            path = OUT / f"{text.replace(' ', '_')}__{stage}.png"
            path.write_bytes(plots.render(stage, tr, params.words, params.emb))
        print(f"\nCharts saved in {OUT}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["cat run", "dog eat"])
