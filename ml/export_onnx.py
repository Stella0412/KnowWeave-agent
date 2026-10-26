from __future__ import annotations

import argparse
import inspect
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class OnnxExportError(RuntimeError):
    pass


def _load(model_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
    return tokenizer, model


def export_onnx(model_dir, out_dir, *, opset=17) -> dict:
    model_dir = Path(model_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer, model = _load(model_dir)
    dummy = tokenizer(["买大了想退", "物流到哪了"], max_length=128, truncation=True,
                      padding="max_length", return_tensors="pt")
    inputs = (dummy["input_ids"], dummy["attention_mask"])
    onnx_path = out_dir / "model.onnx"
    export_kwargs = dict(
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "logits": {0: "batch"},
        },
        opset_version=opset,
        do_constant_folding=True,
    )
    # torch 2.6+ 默认走 dynamo exporter；旧路径与 opset 17 更稳，签名支持时显式关闭
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False
    torch.onnx.export(model, inputs, str(onnx_path), **export_kwargs)
    tokenizer.save_pretrained(out_dir)
    for name in ("thresholds.json", "label_map.json", "manifest.json"):
        source = model_dir.parent / name
        if source.exists():
            shutil.copy2(source, out_dir / name)
    parity = check_parity(model_dir, onnx_path, ["买大了想退", "物流到哪了", "发票怎么开"])
    if parity >= 1e-3:
        raise OnnxExportError(f"ONNX 与 PyTorch 偏差 {parity} 超过 1e-3")
    (out_dir / "export_manifest.json").write_text(json.dumps({
        "source_model": str(model_dir), "opset": opset, "max_abs_diff": parity,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"onnx": str(onnx_path), "max_abs_diff": parity}


def check_parity(model_dir, onnx_path, samples, *, atol=1e-3) -> float:
    import onnxruntime as ort

    tokenizer, model = _load(model_dir)
    encoded = tokenizer(list(samples), max_length=128, truncation=True,
                        padding=True, return_tensors="pt")
    with torch.no_grad():
        torch_logits = model(**encoded).logits.cpu().numpy()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_logits = session.run(None, {
        "input_ids": encoded["input_ids"].numpy(),
        "attention_mask": encoded["attention_mask"].numpy(),
    })[0]
    return float(np.max(np.abs(torch_logits - ort_logits)))


def promote_current(export_dir, current_dir) -> None:
    export_dir = Path(export_dir)
    current_dir = Path(current_dir)
    if current_dir.exists():
        shutil.rmtree(current_dir)
    shutil.copytree(export_dir, current_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="data/ch10/model/latest.txt")
    parser.add_argument("--out", default=None)
    parser.add_argument("--promote-current", default=None)
    args = parser.parse_args()
    model_dir = Path(args.model_dir)
    if model_dir.suffix == ".txt" and model_dir.exists():
        run = model_dir.parent / model_dir.read_text(encoding="utf-8").strip()
        model_dir = run / "best"
        default_out = run / "onnx"
    else:
        default_out = Path(args.out or "data/ch10/model/onnx")
    out = Path(args.out) if args.out else default_out
    result = export_onnx(model_dir, out)
    if args.promote_current:
        promote_current(out, args.promote_current)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
