from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from ml.service.classifier import OnnxTopicClassifier

MODEL_DIR = Path(os.getenv("TOPIC_MODEL_DIR", "data/ch10/model/current"))
MAX_BATCH = int(os.getenv("TOPIC_MAX_BATCH", "128"))
THREADS = int(os.getenv("TOPIC_ONNX_THREADS", "0"))


class ClassifyRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=MAX_BATCH)

    @field_validator("texts", mode="before")
    @classmethod
    def _validate_texts(cls, value):
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise ValueError("texts 必须是非空字符串列表")
        return value


class ClassifyResponse(BaseModel):
    model_version: str
    results: list[list[dict]]


def create_app(classifier=None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if getattr(app.state, "classifier", None) is None:
            app.state.classifier = OnnxTopicClassifier(MODEL_DIR, threads=THREADS)
        yield

    app = FastAPI(title="ch10 topic classifier", lifespan=lifespan)
    if classifier is not None:
        app.state.classifier = classifier

    @app.get("/health")
    def health():
        clf = getattr(app.state, "classifier", None)
        if clf is None:
            raise HTTPException(status_code=503, detail="模型未加载")
        return {"status": "ok", "model_version": clf.version, "labels": len(clf.labels)}

    @app.get("/model")
    def model():
        clf = getattr(app.state, "classifier", None)
        model_dir = getattr(clf, "model_dir", MODEL_DIR)
        path = Path(model_dir) / "export_manifest.json"
        if not path.exists():
            return {"model_dir": str(model_dir)}
        return json.loads(path.read_text(encoding="utf-8"))

    @app.post("/classify", response_model=ClassifyResponse)
    def classify(req: ClassifyRequest):
        clf = getattr(app.state, "classifier", None)
        if clf is None:
            raise HTTPException(status_code=503, detail="模型未加载")
        results = clf.classify(req.texts)
        return ClassifyResponse(model_version=clf.version, results=results)

    return app


app = create_app()
