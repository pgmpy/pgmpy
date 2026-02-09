#!/usr/bin/env python3
"""
Generate docs/datasets.csv from dataset metadata in pgmpy/datasets.
"""

from __future__ import annotations

import ast
import csv
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATASETS_DIR = ROOT_DIR / "pgmpy" / "datasets"
OUTPUT_CSV = ROOT_DIR / "docs" / "datasets.csv"


CSV_COLUMNS = [
    ("name", "Dataset"),
    ("n_variables", "Variables"),
    ("n_samples", "Samples"),
    ("has_ground_truth", "Ground Truth"),
    ("has_expert_knowledge", "Expert Knowledge"),
    ("has_missing_data", "Missing Data"),
    ("is_simulated", "Simulated"),
    ("is_interventional", "Interventional"),
    ("is_discrete", "Discrete"),
    ("is_continuous", "Continuous"),
    ("is_mixed", "Mixed"),
    ("is_ordinal", "Ordinal"),
]

DEFAULT_TAGS = {
    "name": "",
    "n_variables": "",
    "n_samples": "",
    "has_ground_truth": False,
    "has_expert_knowledge": False,
    "has_missing_data": False,
    "is_simulated": False,
    "is_interventional": False,
    "is_discrete": False,
    "is_continuous": False,
    "is_mixed": False,
    "is_ordinal": False,
}


def _is_dataset_class(class_node: ast.ClassDef) -> bool:
    for base in class_node.bases:
        if isinstance(base, ast.Name) and base.id == "_BaseDataset":
            return True

        if isinstance(base, ast.Attribute) and base.attr == "_BaseDataset":
            return True

    return False


def _extract_tags(class_node: ast.ClassDef) -> dict | None:
    for stmt in class_node.body:
        if not isinstance(stmt, ast.Assign):
            continue

        if not any(
            isinstance(target, ast.Name) and target.id == "_tags"
            for target in stmt.targets
        ):
            continue

        try:
            value = ast.literal_eval(stmt.value)
        except (SyntaxError, ValueError):
            return None

        if isinstance(value, dict):
            return value

    return None


def _to_cell(value):
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if value is None:
        return ""
    return value


def generate_rows() -> list[dict]:
    rows = []

    for file_path in sorted(DATASETS_DIR.glob("*.py")):
        if file_path.name.startswith("_"):
            continue

        tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))

        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            if not _is_dataset_class(node):
                continue

            tags = _extract_tags(node)
            if not tags or not tags.get("name"):
                continue

            row = {
                column_name: _to_cell(tags.get(tag_name, DEFAULT_TAGS[tag_name]))
                for tag_name, column_name in CSV_COLUMNS
            }
            rows.append(row)

    rows.sort(key=lambda row: str(row["Dataset"]))
    return rows


def main() -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    rows = generate_rows()
    fieldnames = [column_name for _, column_name in CSV_COLUMNS]

    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} datasets to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
