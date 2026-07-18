"""Shared output location for newly run simulations."""

from pathlib import Path

import pandas as pd


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "results" / "simulations" / "generated"


def save_results(frame: pd.DataFrame, filename: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / filename
    frame.to_parquet(path, index=False)
    return path
