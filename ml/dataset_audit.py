"""审计用户提供的 ch10-dataset：只读校验、统计、manifest，不改写数据。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path

from ml.taxonomy import Taxonomy, load_taxonomy

SPLITS = ("train", "val", "test")

_PII_PATTERNS = (
    re.compile(r"(?<![\d.])1[3-9]\d{9}(?![\d.])"),
    re.compile(r"(?<![\d.])\d{17}[\dXx](?![\d.])"),
    re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b"),
)


class DatasetError(ValueError):
    pass


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise DatasetError(f"数据集文件不存在: {path}") from exc
    rows = []
    for lineno, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"{path.name}:{lineno} 不是合法 JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise DatasetError(f"{path.name}:{lineno} 必须是 JSON 对象")
        rows.append(row)
    return rows


def load_split(path: str | Path) -> list[dict]:
    path = Path(path)
    rows = []
    for lineno, row in enumerate(read_jsonl(path), 1):
        text = row.get("text")
        if not isinstance(text, str) or not text.strip():
            raise DatasetError(f"{path.name}:{lineno} text 必须是非空字符串")
        labels = row.get("labels")
        if not isinstance(labels, list) or not labels:
            raise DatasetError(f"{path.name}:{lineno} labels 不能为空")
        if not all(isinstance(label, str) and label.strip() for label in labels):
            raise DatasetError(f"{path.name}:{lineno} labels 必须是非空字符串列表")
        if len(set(labels)) != len(labels):
            raise DatasetError(f"{path.name}:{lineno} labels 有重复项")
        rows.append({"text": text.strip(), "labels": list(labels)})
    if not rows:
        raise DatasetError(f"{path.name} 没有有效样本")
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pii_hits(text: str) -> int:
    return sum(1 for pattern in _PII_PATTERNS if pattern.search(text))


def audit_dataset(root: str | Path, taxonomy: Taxonomy | None = None) -> dict:
    root = Path(root)
    taxonomy = taxonomy or load_taxonomy()
    names = taxonomy.names
    name_set = set(names)

    split_rows: dict[str, list[dict]] = {}
    for split in SPLITS:
        split_rows[split] = load_split(root / f"{split}.jsonl")

    for split, rows in split_rows.items():
        for index, row in enumerate(rows, 1):
            for label in row["labels"]:
                if label not in name_set:
                    raise DatasetError(f"{split}:{index} 未知标签: {label}")

    text_sets = {split: {row["text"] for row in rows} for split, rows in split_rows.items()}
    for i, left in enumerate(SPLITS):
        for right in SPLITS[i + 1:]:
            overlap = text_sets[left] & text_sets[right]
            if overlap:
                sample = next(iter(sorted(overlap)))
                raise DatasetError(
                    f"跨 split 重复文本 {left}/{right}: {len(overlap)} 条，例如 {sample!r}"
                )

    warnings: list[str] = []
    duplicates_within: dict[str, int] = {}
    pii_hits = 0
    total_rows = 0
    total_multi = 0
    total_other_coexist = 0
    total_labels = {name: 0 for name in names}
    split_stats = {}
    for split in SPLITS:
        rows = split_rows[split]
        counts = {name: 0 for name in names}
        multi = 0
        for row in rows:
            if len(row["labels"]) > 1:
                multi += 1
            if "其他" in row["labels"] and len(row["labels"]) > 1:
                total_other_coexist += 1
            for label in row["labels"]:
                counts[label] += 1
                total_labels[label] += 1
            pii_hits += _pii_hits(row["text"])
        seen: dict[str, int] = {}
        for row in rows:
            seen[row["text"]] = seen.get(row["text"], 0) + 1
        duplicate_count = sum(1 for count in seen.values() if count > 1)
        duplicates_within[split] = duplicate_count
        if duplicate_count:
            warnings.append(f"{split} 有 {duplicate_count} 组完全重复文本")
        low_support = [f"{name}={count}" for name, count in counts.items() if count < 40]
        if low_support:
            warnings.append(f"{split} 每类 support < 40: {', '.join(low_support)}")
        path = root / f"{split}.jsonl"
        split_stats[split] = {
            "rows": len(rows),
            "multi_label": multi,
            "labels": counts,
            "sha256": _sha256(path),
        }
        total_rows += len(rows)
        total_multi += multi

    if pii_hits:
        warnings.append(f"检测到 {pii_hits} 条疑似 PII，训练前需脱敏")

    return {
        "version": 1,
        "taxonomy_version": taxonomy.version,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "root": str(root),
        "splits": split_stats,
        "totals": {
            "rows": total_rows,
            "multi_label": total_multi,
            "other_coexist": total_other_coexist,
            "labels": total_labels,
            "pii_hits": pii_hits,
        },
        "duplicates_within": duplicates_within,
        "pii_hits": pii_hits,
        "warnings": warnings,
    }


def write_manifest(root: str | Path, out: str | Path | None = None) -> Path:
    root = Path(root)
    manifest = audit_dataset(root)
    path = Path(out) if out is not None else root / "manifest.json"
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="审计 ch10-dataset 并生成 manifest")
    parser.add_argument("--root", default="ch10-dataset")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    try:
        path = write_manifest(args.root, args.out)
    except DatasetError as exc:
        print(f"数据集审计失败: {exc}", file=sys.stderr)
        return 1
    manifest = json.loads(path.read_text(encoding="utf-8"))
    print(
        f"审计完成: {manifest['totals']['rows']} 条，"
        f"多标签 {manifest['totals']['multi_label']} 条，"
        f"PII {manifest['pii_hits']} 条；manifest: {path}"
    )
    for warning in manifest["warnings"]:
        print(f"  warning: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
