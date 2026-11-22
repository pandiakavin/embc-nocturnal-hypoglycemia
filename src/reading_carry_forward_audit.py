"""Audit exact repeated glucose values across adjacent observed 5-minute readings.

This is a provenance diagnostic, not proof of forward filling: real CGM values
can remain unchanged. It counts only pairs from the same subject whose timestamps
are exactly five minutes apart and whose glucose values are both observed.

Usage:
  python src/reading_carry_forward_audit.py --main data/raw/MDB_5min_sampling_raw.csv
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd


def audit(path: Path, chunksize: int = 500_000) -> pd.DataFrame:
    usecols = ["ts", "PtID", "GlucoseCGM", "Database"]
    counts = defaultdict(lambda: [0, 0])  # source -> [valid adjacent pairs, exact repeats]
    # The raw export is grouped by subject. Keep the last observation so pairs
    # crossing a CSV chunk boundary are handled without loading the full file.
    last: dict[str, tuple[pd.Timestamp, float, str]] = {}
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize,
                             dtype={"PtID": str, "Database": str}):
        chunk["ts"] = pd.to_datetime(chunk["ts"], errors="coerce")
        chunk["GlucoseCGM"] = pd.to_numeric(chunk["GlucoseCGM"], errors="coerce")
        if chunk["ts"].isna().any():
            raise ValueError("Found invalid timestamps in main database")
        for pid, g in chunk.groupby("PtID", sort=False):
            g = g.reset_index(drop=True)
            if not g["ts"].is_monotonic_increasing:
                raise ValueError(f"Timestamps are not ordered for subject {pid}")
            sources = g["Database"].dropna().unique()
            if len(sources) != 1:
                raise ValueError(f"Subject {pid} maps to multiple source datasets")
            src = str(sources[0])
            ts = g["ts"]
            glucose = g["GlucoseCGM"]
            prev_ts = ts.shift(1)
            prev_glucose = glucose.shift(1)
            prior = last.get(str(pid))
            if prior is not None:
                # A decreasing timestamp across chunks would invalidate the audit.
                if ts.iloc[0] < prior[0]:
                    raise ValueError(f"Timestamps went backwards across chunks for {pid}")
                prev_ts.iloc[0] = prior[0]
                prev_glucose.iloc[0] = prior[1]
            adjacent = (ts - prev_ts).eq(pd.Timedelta(minutes=5))
            valid = adjacent & glucose.notna() & prev_glucose.notna()
            counts[src][0] += int(valid.sum())
            counts[src][1] += int((valid & glucose.eq(prev_glucose)).sum())
            last[str(pid)] = (ts.iloc[-1], float(glucose.iloc[-1]) if pd.notna(glucose.iloc[-1]) else float("nan"), src)
    rows = []
    for src, (n_pairs, n_repeated) in sorted(counts.items()):
        rows.append({
            "source_dataset": src,
            "observed_adjacent_5min_pairs": n_pairs,
            "exactly_repeated_pairs": n_repeated,
            "pct_exactly_repeated": (100 * n_repeated / n_pairs) if n_pairs else float("nan"),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--main", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=Path("results/reading_carry_forward_audit.csv"))
    ap.add_argument("--chunksize", type=int, default=500_000)
    args = ap.parse_args()
    result = audit(args.main, args.chunksize)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.round(4).to_csv(args.output, index=False)
    print(result.round(2).to_string(index=False))
    print(f"\nSaved {args.output}")
    print("Interpretation: repeated values may reflect sensor plateaus; this audit alone cannot prove imputation.")


if __name__ == "__main__":
    main()
