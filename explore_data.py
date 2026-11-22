"""Prompt 3 - explore DiaData (read-only).

Reports, overall and per age group:
  * number of subjects
  * days of data per subject (calendar days that have >=1 reading, and total span)
  * % missing readings = 1 - (5-min slots with a glucose value) / (slots between
    the subject's first and last reading)
  * how many subjects have an exact age vs. only an age range

Usage (from the project folder):
  python src/explore_data.py --main data/raw/maindatabase.csv \
      --demographics data/raw/subdatabaseI.csv --age-scheme broad

Outputs:
  results/data_summary.csv              one row per age group + "All"
  results/data_summary_by_database.csv  one row per source dataset inside DiaData
  results/subject_summary.csv           one row per subject (for checking)
  results/raw_checksums.txt             SHA-256 of the raw files you used
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import diadata_io as io


def per_subject_table(main: dict) -> pd.DataFrame:
    rows = []
    for c, a, b in io.split_by_subject(main["code"]):
        slot = main["slot"][a:b]
        glu = main["glu"][a:b]
        u, mean, n_dup = io.collapse_duplicate_slots(slot, glu)
        present = ~np.isnan(mean)
        expected = int(u[-1] - u[0] + 1)
        days_with_data = int(np.unique(u[present] // io.SLOTS_PER_DAY).size)
        rows.append({
            io.PID: main["pids"][c],
            "source_dataset": main["db"][c],
            "first_ts": pd.Timestamp("1970-01-01") + pd.Timedelta(minutes=5 * int(u[0])),
            "last_ts": pd.Timestamp("1970-01-01") + pd.Timedelta(minutes=5 * int(u[-1])),
            "span_days": round(expected / io.SLOTS_PER_DAY, 2),
            "days_with_data": days_with_data,
            "n_rows": int(b - a),
            "n_rows_glucose_missing": int(np.isnan(glu).sum()),
            "n_rows_sharing_a_slot": n_dup,
            "n_slots_present": int(present.sum()),
            "n_slots_expected": expected,
            "missing_pct": round(100 * (1 - present.sum() / expected), 2),
        })
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame, by: str) -> pd.DataFrame:
    def block(g: pd.DataFrame) -> dict:
        q = g["days_with_data"].quantile([0.25, 0.5, 0.75])
        return {
            "n_subjects": len(g),
            "days_with_data_median": q[0.5],
            "days_with_data_q1": q[0.25],
            "days_with_data_q3": q[0.75],
            "days_with_data_total": int(g["days_with_data"].sum()),
            "span_days_median": g["span_days"].median(),
            "missing_pct_median_subject": g["missing_pct"].median(),
            # pooled = all missing slots / all expected slots in the group
            "missing_pct_pooled": round(
                100 * (1 - g["n_slots_present"].sum() / g["n_slots_expected"].sum()), 2),
            "n_age_exact": int((g["age_type"] == "exact").sum()),
            "n_age_range_only": int((g["age_type"] == "range").sum()),
            "n_age_missing": int((g["age_type"] == "missing").sum()),
        }
    out = [{by: "All", **block(df)}]
    for key, g in df.groupby(by, sort=True):
        out.append({by: key, **block(g)})
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--main", type=Path, required=True, help="DiaData main database CSV")
    ap.add_argument("--demographics", type=Path, default=None,
                    help="DiaData sub-database CSV with Age/AgeGroup columns")
    ap.add_argument("--age-scheme", default="kids_teens_adults", choices=io.AGE_SCHEMES)
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--chunksize", type=int, default=2_000_000)
    ap.add_argument("--no-checksum", action="store_true")
    args = ap.parse_args()

    io.check_output_dir(args.out, [args.main, args.demographics])
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Reading {args.main} (read-only) ...")
    main_arr = io.load_main_arrays(args.main, args.chunksize)
    subj = per_subject_table(main_arr)

    demo = io.load_demographics(args.demographics, args.chunksize)
    subj = subj.merge(demo, how="left", left_on=io.PID, right_index=True)
    subj["age_group"] = (io.age_labels(demo, args.age_scheme)
                         .reindex(subj[io.PID]).fillna(io.UNKNOWN).to_numpy()
                         if not demo.empty else io.UNKNOWN)
    subj["age_type"] = (subj[io.AGE].map(io.age_type) if io.AGE in subj.columns
                        else "missing")

    summary = summarise(subj, "age_group")
    by_db = summarise(subj, "source_dataset")
    summary.to_csv(args.out / "data_summary.csv", index=False)
    by_db.to_csv(args.out / "data_summary_by_database.csv", index=False)
    subj.to_csv(args.out / "subject_summary.csv", index=False)

    if not args.no_checksum:
        with open(args.out / "raw_checksums.txt", "w") as f:
            for p in (args.main, args.demographics):
                if p is not None:
                    f.write(f"{io.sha256(p)}  {p}\n")

    # ---- plain-English report ----
    n_demo_conflicts = 0
    print(f"\nSubjects in main database: {len(subj)}")
    print(f"Subjects with demographics: {int(subj[io.PID].isin(demo.index).sum())}")
    if "n_distinct_age_values" in subj.columns:
        print(f"Subjects whose Age changes during follow-up (age at first reading used): "
              f"{int((subj['n_distinct_age_values'] > 1).sum())}")
    if "age_low" in subj.columns:
        print(f"Subjects with an age range that fits no single group (unknown): "
              f"{int(((subj['age_type'] == 'range') & (subj['age_group'] == io.UNKNOWN)).sum())}")
    if io.AGE not in demo.columns:
        print("NOTE: no 'Age' column in the demographics file, so exact-vs-range "
              "cannot be checked (all counted as 'missing').")
    if n_demo_conflicts:
        print(f"WARNING: {n_demo_conflicts} subjects have >1 distinct demographic row "
              "(e.g. age group changed during a multi-year study). First row was used.")
    print(f"Rows that shared a 5-min slot (averaged): {int(subj['n_rows_sharing_a_slot'].sum())}")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print("\nSummary by age group:\n", summary.to_string(index=False))
        print("\nSummary by source dataset:\n", by_db.to_string(index=False))
    print(f"\nSaved to {args.out}/")


if __name__ == "__main__":
    main()
