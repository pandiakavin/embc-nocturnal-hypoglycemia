"""Summarize source-dataset composition by age group and split.

Usage: python src/source_composition.py [--tag n00-06_lt70_r4]
Writes results/source_composition_<tag>.csv. Reads feature tables only.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PID = "PtID"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="n00-06_lt70_r4")
    ap.add_argument("--results", type=Path, default=Path("results"))
    args = ap.parse_args()
    parts = []
    for split in ("train", "val", "test"):
        p = args.results / "features" / f"features_{args.tag}_{split}.csv"
        d = pd.read_csv(p, usecols=[PID, "age_group", "source_dataset", "label"])
        d["split"] = split
        parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    out = (d.groupby(["split", "age_group", "source_dataset"], observed=True)
             .agg(n_subjects=(PID, "nunique"), n_nights=("label", "size"),
                  n_positive=("label", "sum"), prevalence=("label", "mean"))
             .reset_index())
    out["prevalence"] = out["prevalence"].round(4)
    path = args.results / f"source_composition_{args.tag}.csv"
    out.to_csv(path, index=False)
    print(out[out["split"] == "test"].to_string(index=False))
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
