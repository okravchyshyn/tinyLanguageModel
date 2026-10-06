# Tiny LLM next-token demo

Educational walk-through of how a Transformer predicts the next token. Not for performance or production.

## Run it

```bash
pip install -r requirements.txt
python examples/build_model.py          # corpus -> vocabulary + embeddings in database/llm_demo.db
python examples/run_demo.py             # "cat run" and "dog eat" step by step, PNG charts in examples/output/
uvicorn src.api.main:app --reload       # API docs at http://127.0.0.1:8000/docs
pytest
```

The API builds the model on first start if `data/processed/model.npz` is missing.

## Pipeline and endpoints

| Step | Endpoint | Table(s) |
|---|---|---|
| Vocabulary | `GET /vocabulary` | `vocabulary` |
| Tokenize | `POST /tokenize` | - |
| Embeddings | `POST /embeddings` | `embeddings` |
| Q/K/V | `POST /attention/qkv` | `queries`, `keys`, `values` |
| Scores `Q.K^T / sqrt(d)` | `POST /attention/scores` | `attention_scores` |
| Softmax | `POST /attention/matrix` | `attention_matrices` |
| Attention x V | `POST /attention/weighted-values` | `weighted_values` |
| Hidden state H | `POST /hidden-state` | `hidden_states` |
| Next token | `POST /predict` | `predictions` |
| Everything | `POST /explain` (`include_plots` for base64 PNGs) | all |
| Charts | `POST /visualize/{embeddings,attention-scores,attention-matrix,hidden-state,predictions}` | - |

Request body: `{"text": "cat run", "causal": true, "top_k": 5, "similar_k": 5}`.
Each stage response has `math_view`, `human_view` and `visual_view`.

## How the weights are made

- Vocabulary: 550 most frequent words (so that "fast" is included) plus `<unk>` = id 0; `id == rank`.
- Embeddings (16-d): co-occurrence counts -> PPMI -> SVD, so words in similar contexts are close.
- `Wq`, `Wk`, `Wv`, `Wout` + bias: start from seeded random / bigram-fitted values, then are trained together
  (Adam, hand-written backprop, causal next-token loss on random 6-word windows; embeddings stay frozen).
  Q/K/V role descriptions are hand-written hints.
- Limits: no positional encoding, multi-head attention or extra layers, so it stays a very small model.
