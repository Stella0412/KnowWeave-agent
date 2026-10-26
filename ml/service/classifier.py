from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from ml.decision import decide


class OnnxTopicClassifier:
    def __init__(self, model_dir, session=None, tokenizer=None, threads: int = 0):
        self._dir = Path(model_dir)
        if session is None:
            options = ort.SessionOptions()
            if threads:
                options.intra_op_num_threads = threads
            session = ort.InferenceSession(
                str(self._dir / "model.onnx"), options, providers=["CPUExecutionProvider"],
            )
        self._session = session
        self._tokenizer = tokenizer or self._load_tokenizer()
        label_map = json.loads((self._dir / "label_map.json").read_text(encoding="utf-8"))
        self._names = tuple(label_map["labels"])
        thresholds = json.loads((self._dir / "thresholds.json").read_text(encoding="utf-8"))
        self._thresholds = [float(thresholds["thresholds"][name]) for name in self._names]
        manifest_path = self._dir / "manifest.json"
        self._version = "unknown"
        if manifest_path.exists():
            self._version = json.loads(manifest_path.read_text(encoding="utf-8")).get(
                "run", self._dir.name
            )

    def _load_tokenizer(self):
        tokenizer_json = self._dir / "tokenizer.json"
        if tokenizer_json.exists():
            from tokenizers import Tokenizer
            tokenizer = Tokenizer.from_file(str(tokenizer_json))
            tokenizer.enable_truncation(max_length=128)
            tokenizer.enable_padding(length=128)
            return _TokenizerAdapter(tokenizer)
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(self._dir)

    @property
    def version(self) -> str:
        return self._version

    @property
    def labels(self) -> tuple[str, ...]:
        return self._names

    @property
    def model_dir(self) -> Path:
        return self._dir

    def classify(self, texts):
        if not texts:
            return []
        if any(not isinstance(text, str) for text in texts):
            raise ValueError("texts 必须全是字符串")
        encoded = self._tokenizer(list(texts), max_length=128, truncation=True,
                                  padding=True, return_tensors="np")
        logits = self._session.run(None, {
            "input_ids": encoded["input_ids"].astype(np.int64),
            "attention_mask": encoded["attention_mask"].astype(np.int64),
        })[0]
        probs = 1 / (1 + np.exp(-logits))
        out = []
        for row in probs:
            labels = decide([float(x) for x in row], self._thresholds, self._names)
            scores = {name: float(row[i]) for i, name in enumerate(self._names)}
            out.append([{"label": name, "score": round(scores[name], 4)} for name in labels])
        return out


class _TokenizerAdapter:
    def __init__(self, tokenizer):
        self._tokenizer = tokenizer

    def __call__(self, texts, max_length, truncation, padding, return_tensors="np"):
        encodings = self._tokenizer.encode_batch(list(texts))
        return {
            "input_ids": np.array([enc.ids for enc in encodings], dtype=np.int64),
            "attention_mask": np.array([enc.attention_mask for enc in encodings], dtype=np.int64),
        }
