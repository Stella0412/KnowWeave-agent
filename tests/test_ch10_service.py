import numpy as np
import pytest

pytest.importorskip("onnxruntime")
from fastapi.testclient import TestClient  # noqa: E402

from ml.service.app import create_app  # noqa: E402
from ml.service.classifier import OnnxTopicClassifier  # noqa: E402


class FakeTokenizer:
    def __call__(self, texts, max_length, truncation, padding, return_tensors="np"):
        return {"input_ids": np.ones((len(texts), 4), dtype=np.int64),
                "attention_mask": np.ones((len(texts), 4), dtype=np.int64)}


class FakeSession:
    def run(self, output_names, inputs):
        return [np.array([[5.0] + [-5.0] * 16 for _ in range(len(inputs["input_ids"]))])]


def _classifier():
    return OnnxTopicClassifier.__new__(OnnxTopicClassifier)


def test_classifier_returns_labels_and_scores():
    clf = _classifier()
    clf._tokenizer = FakeTokenizer()
    clf._session = FakeSession()
    clf._names = tuple(["退换货"] + [f"类{i}" for i in range(15)] + ["其他"])
    clf._thresholds = [0.5] * 17
    clf._version = "test"
    out = clf.classify(["买大了想退"])
    assert out[0][0]["label"] == "退换货"
    assert 0.0 <= out[0][0]["score"] <= 1.0


class FakeClassifier:
    version = "test"
    labels = ("退换货",) + tuple(f"类{i}" for i in range(15)) + ("其他",)

    def classify(self, texts):
        return [[{"label": "其他", "score": 0.9}] for _ in texts]


def test_classify_endpoint_validates_batch_size():
    app = create_app(classifier=FakeClassifier())
    client = TestClient(app)
    assert client.get("/health").status_code == 200
    too_many = {"texts": ["x"] * 129}
    assert client.post("/classify", json=too_many).status_code == 422
    assert client.post("/classify", json={"texts": []}).status_code == 422
    assert client.post("/classify", json={"texts": ["买大了想退"]}).status_code == 200
    assert client.post("/classify", json={"texts": [123]}).status_code == 422
