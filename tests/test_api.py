import pytest


def test_vocabulary(client):
    r = client.get("/vocabulary", params={"limit": 3})
    assert r.status_code == 200 and {"token", "frequency", "rank"} <= set(r.json()[0])


def test_tokenize(client):
    body = client.post("/tokenize", json={"text": "cat run"}).json()
    assert body["tokens"] == ["cat", "run"] and len(body["ids"]) == 2


@pytest.mark.parametrize(
    "path,key",
    [
        ("/embeddings", "embeddings"),
        ("/attention/qkv", "human_explanation"),
        ("/attention/scores", "raw_scores"),
        ("/attention/matrix", "attention"),
        ("/attention/weighted-values", "weighted_values"),
        ("/hidden-state", "hidden_state"),
        ("/predict", "top_predictions"),
    ],
)
def test_stage_endpoints(client, path, key):
    r = client.post(path, json={"text": "cat run"})
    assert r.status_code == 200 and key in r.json()


def test_explain_and_plots(client):
    r = client.post("/explain", json={"text": "dog eat", "include_plots": True})
    body = r.json()
    assert {"tokenization", "embeddings", "qkv", "attention", "hidden_state", "predictions"} <= set(body)
    assert len(body["plots_base64_png"]) == 5


def test_visualize_png_and_errors(client):
    r = client.post("/visualize/attention-matrix", json={"text": "cat run"})
    assert r.headers["content-type"] == "image/png" and r.content[:4] == b"\x89PNG"
    assert client.post("/visualize/nope", json={"text": "cat"}).status_code == 404
    assert client.post("/predict", json={"text": "!!!"}).status_code == 422
