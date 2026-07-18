"""Convert scored generation JSONL files to LaRT input matrices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()

    rows = []
    for path in args.inputs:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    if record.get("correct") not in (0, 1, False, True):
                        raise ValueError(f"{path}: every record needs a verified 0/1 'correct' field")
                    rows.append(record)

    frame = pd.DataFrame(rows)
    required = {"model", "prompt", "id", "correct", "cot_tokens"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing required fields: {sorted(missing)}")
    frame["row_id"] = frame["model"].str.replace(r"[^A-Za-z0-9]+", "_", regex=True) + "_" + frame["prompt"]
    accuracy = frame.pivot(index="row_id", columns="id", values="correct").sort_index(axis=1)
    latency = frame.pivot(index="row_id", columns="id", values="cot_tokens").sort_index(axis=1)
    if accuracy.isna().any().any() or latency.isna().any().any():
        raise ValueError("the model/prompt by item grid is incomplete")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    accuracy.astype(int).to_csv(args.output_dir / f"correctness_matrix_{args.dataset}.csv")
    latency.astype(int).to_csv(args.output_dir / f"cot_length_matrix_{args.dataset}.csv")


if __name__ == "__main__":
    main()
