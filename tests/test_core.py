import numpy as np
import pytest

from src.attention import attention as att
from src.pipeline import EmptyTextError, Pipeline
from src.tokenization.vocabulary import build_vocabulary, split_words


def test_vocabulary_ranks_and_unk():
    entries = build_vocabulary(["a", "b", "a", "c", "a", "b"], size=2)
    assert [(e.word, e.frequency, e.rank, e.token_id) for e in entries] == [
        ("<unk>", 1, 0, 0), ("a", 3, 1, 1), ("b", 2, 2, 2)
    ]


def test_split_words_lowercases_and_drops_punctuation():
    assert split_words("Cat, run! don't") == ["cat", "run", "don't"]


def test_softmax_rows_sum_to_one_and_causal_mask():
    rng = np.random.default_rng(0)
    a, mask = att.attention_weights(rng.normal(size=(3, 3)), causal=True)
    assert np.allclose(a.sum(axis=1), 1)
    assert np.all(a[mask] == 0)


def test_attention_times_v_equals_sum_of_contributions():
    rng = np.random.default_rng(1)
    a = att.softmax(rng.normal(size=(3, 3)))
    v = rng.normal(size=(3, 4))
    contributions, h = att.weighted_values(a, v)
    assert np.allclose(h, a @ v)


def test_pipeline_shapes_and_persistence(built):
    _, engine, params = built
    pipe = Pipeline(params, engine)
    tr = pipe.run("cat run")
    assert tr.q.shape == tr.k.shape == tr.v.shape == (2, params.dim)
    assert tr.attn.shape == (2, 2) and np.isclose(tr.probs.sum(), 1)
    assert tr.top[0]["token"] != "<unk>"
    from sqlalchemy import text

    with engine.connect() as c:
        n = c.execute(text("select count(*) from predictions where run_id = :r"), {"r": tr.run_id}).scalar()
    assert n == 5


def test_tokenization_reports_coverage_and_generation(built):
    _, engine, params = built
    pipe = Pipeline(params, engine)
    tr = pipe.trace("cat mysteryword")
    tok = pipe.tokenization(tr)
    assert tok["coverage"]["unknown_token_count"] == 1
    assert 0 <= tok["coverage"]["vocabulary_coverage"] < 1

    gen = pipe.generate("cat run", steps=2)
    assert gen["steps"] == 2
    assert len(gen["generated"]) == 2
    assert isinstance(gen["final_text"], str) and gen["final_text"]


def test_empty_text_rejected(built):
    _, engine, params = built
    with pytest.raises(EmptyTextError):
        Pipeline(params, engine).trace("!!!")
