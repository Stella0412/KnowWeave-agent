import argparse
from pathlib import Path

from ml.evaluate import evaluate_model


def _resolve_model_dir(value: str) -> str:
    path = Path(value)
    if path.suffix == ".txt" and path.exists():
        return str(path.parent / path.read_text(encoding="utf-8").strip() / "best")
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="data/ch10/model/latest.txt")
    parser.add_argument("--test", default="ch10-dataset/test.jsonl")
    parser.add_argument("--real", default="ch10-dataset/real_slice.jsonl")
    parser.add_argument("--golden", default="eval/ch10_golden_cases.jsonl")
    parser.add_argument("--out", default="data/ch10/eval")
    parser.add_argument("--report", default="eval/reports/ch10-topic-2026-10-26.md")
    args = parser.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    result = evaluate_model(_resolve_model_dir(args.model_dir), args.test, args.real,
                            args.golden, args.out, args.report)
    print(result["meta"])
    print(result["metrics"])


if __name__ == "__main__":
    main()
