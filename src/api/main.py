"""FastAPI backend: one endpoint per stage plus /explain for the whole report.

Run from the project root:  uvicorn src.api.main:app --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, Field

from src import config
from src.database import make_engine
from src.model import ModelParams, build_model
from src.pipeline import EmptyTextError, Pipeline
from src.tokenization.vocabulary import load_vocabulary
from src.visualization import plots


class TextRequest(BaseModel):
    text: str = Field(min_length=1, max_length=500, examples=["cat run"])
    causal: bool = True
    top_k: int = Field(5, ge=1, le=20)
    similar_k: int = Field(5, ge=1, le=20)


class ExplainRequest(TextRequest):
    include_plots: bool = False


def create_app(db_path: Optional[Path] = None, model_path: Optional[Path] = None) -> FastAPI:
    db_path = db_path or config.DB_PATH
    model_path = model_path or config.MODEL_PATH

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        engine = make_engine(db_path)
        if model_path.exists():
            params = ModelParams.load(model_path)
        else:
            params = build_model(engine, model_path=model_path)
        app.state.engine = engine
        app.state.pipeline = Pipeline(params, engine)
        yield

    app = FastAPI(title="Tiny LLM next-token demo", lifespan=lifespan)

    def run(request: Request, body: TextRequest, upto: str):
        try:
            return request.app.state.pipeline, request.app.state.pipeline.run(
                body.text, upto=upto, causal=body.causal, top_n=body.top_k
            )
        except EmptyTextError as e:
            raise HTTPException(422, str(e))

    @app.get("/vocabulary")
    def vocabulary(request: Request, limit: Optional[int] = None):
        return [
            {"token": e.word, "frequency": e.frequency, "rank": e.rank, "token_id": e.token_id}
            for e in load_vocabulary(request.app.state.engine, limit)
        ]

    @app.post("/tokenize")
    def tokenize(request: Request, body: TextRequest):
        pipe, tr = run(request, body, "tokenize")
        return pipe.tokenization(tr)

    @app.post("/embeddings")
    def embeddings(request: Request, body: TextRequest):
        pipe, tr = run(request, body, "embeddings")
        return pipe.embeddings(tr, body.similar_k)

    @app.post("/attention/qkv")
    def qkv(request: Request, body: TextRequest):
        pipe, tr = run(request, body, "qkv")
        return pipe.qkv(tr)

    @app.post("/attention/scores")
    def scores(request: Request, body: TextRequest):
        pipe, tr = run(request, body, "scores")
        return pipe.scores(tr)

    @app.post("/attention/matrix")
    def matrix(request: Request, body: TextRequest):
        pipe, tr = run(request, body, "matrix")
        return pipe.matrix(tr)

    @app.post("/attention/weighted-values")
    def weighted_values(request: Request, body: TextRequest):
        pipe, tr = run(request, body, "weighted")
        return pipe.weighted(tr)

    @app.post("/hidden-state")
    def hidden_state(request: Request, body: TextRequest):
        pipe, tr = run(request, body, "hidden")
        return pipe.hidden_state(tr)

    @app.post("/predict")
    def predict(request: Request, body: TextRequest):
        pipe, tr = run(request, body, "predict")
        return pipe.prediction(tr)

    @app.post("/explain")
    def explain(request: Request, body: ExplainRequest):
        pipe, tr = run(request, body, "predict")
        report = pipe.explain(tr, body.similar_k)
        if body.include_plots:
            import base64

            p = pipe.p
            report["plots_base64_png"] = {
                s: base64.b64encode(plots.render(s, tr, p.words, p.emb)).decode() for s in plots.STAGES
            }
        return report

    @app.post("/visualize/{stage}", responses={200: {"content": {"image/png": {}}}})
    def visualize(request: Request, stage: str, body: TextRequest):
        if stage not in plots.STAGES:
            raise HTTPException(404, f"unknown stage; choose one of {plots.STAGES}")
        pipe, tr = run(request, body, "predict")
        return Response(plots.render(stage, tr, pipe.p.words, pipe.p.emb), media_type="image/png")

    return app


app = create_app()
