"""Exploratory sensitivity for age-at-first-reading versus age-at-night strata.

Uses the already-fitted global HGB test predictions. It does not retrain models.
For each eligible patient-night, age is taken from the earliest available age
record in dataset-clock [00:00, 06:00). Bootstrap uncertainty is subject-level
and uses the same group-derived deterministic streams as error_analysis.py.

Run from the project root:
  .venv/bin/python src/age_at_night_sensitivity.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import diadata_io as io
from error_analysis import primary_age_gap

PID = io.PID


def age_label(value: object) -> str:
    parsed = io._parse_age(value)
    if parsed is None:
        return io.UNKNOWN
    lo, hi = parsed
    return io._bin_label(lo, hi, io.AGE_BINS["kids_teens_adults"])


def night_age_lookup(demographics: Path, features: pd.DataFrame,
                     chunksize: int) -> tuple[dict[tuple[str, str], tuple[str, str]], int]:
    """Return (PtID, night_date) -> (first timestamp in night, age string)."""
    needed_ids = set(features[PID].astype(str))
    wanted_dates = set(features["night_date"].astype(str))
    wanted_pairs = set(zip(features[PID].astype(str), features["night_date"].astype(str)))
    lookup: dict[tuple[str, str], tuple[str, str]] = {}

    for chunk in pd.read_csv(demographics, usecols=[PID, "ts", "Age"],
                             dtype="string", chunksize=chunksize):
        chunk = chunk[chunk[PID].isin(needed_ids)].dropna(subset=["ts", "Age"])
        if chunk.empty:
            continue
        chunk["night_date"] = chunk["ts"].str.slice(0, 10)
        chunk = chunk[chunk["night_date"].isin(wanted_dates)]
        if chunk.empty:
            continue
        clock = chunk["ts"].str.slice(11, 16)
        chunk = chunk[(clock >= "00:00") & (clock < "06:00")]
        if chunk.empty:
            continue
        chunk["key"] = list(zip(chunk[PID].astype(str), chunk["night_date"].astype(str)))
        chunk = chunk[chunk["key"].isin(wanted_pairs)]
        if chunk.empty:
            continue
        chunk = chunk.sort_values("ts").drop_duplicates("key", keep="first")
        for key, ts, age in chunk[["key", "ts", "Age"]].itertuples(index=False, name=None):
            old = lookup.get(key)
            value = (str(ts), str(age))
            if old is None or value[0] < old[0]:
                lookup[key] = value

    missing = len(wanted_pairs - set(lookup))
    return lookup, missing


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--demographics", type=Path,
                    default=Path("data/raw/SDBI_5min_sampling_raw.csv"))
    ap.add_argument("--features", type=Path,
                    default=Path("results/features/features_n00-06_lt70_r4_test.csv"))
    ap.add_argument("--predictions", type=Path,
                    default=Path("results/baseline_test_predictions_n00-06_lt70_r4.csv"))
    ap.add_argument("--primary-result", type=Path,
                    default=Path("results/error_analysis/n00-06_lt70_r4/primary_auroc_difference.csv"))
    ap.add_argument("--out", type=Path,
                    default=Path("results/age_at_night_sensitivity"))
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chunksize", type=int, default=1_000_000)
    args = ap.parse_args()

    features = pd.read_csv(args.features, dtype={PID: str})
    lookup, missing = night_age_lookup(args.demographics, features, args.chunksize)
    if missing:
        raise SystemExit(f"No in-night age record found for {missing} eligible test nights.")

    features["age_value_at_night"] = [
        lookup[(str(pid), str(date))][1]
        for pid, date in zip(features[PID], features["night_date"])
    ]
    features["age_group_at_night"] = features["age_value_at_night"].map(age_label)
    if (features["age_group_at_night"] == io.UNKNOWN).any():
        n = int((features["age_group_at_night"] == io.UNKNOWN).sum())
        raise SystemExit(f"{n} test nights have an age value that cannot be assigned to one group.")

    predictions = pd.read_csv(args.predictions, dtype={PID: str})
    predictions = predictions[(predictions["model"] == "hgb") &
                              (predictions["scope"] == "global")]
    if predictions["night_id"].duplicated().any():
        raise SystemExit("Global HGB predictions contain duplicate night_id values.")
    joined = features.merge(predictions[["night_id", PID, "label", "prob"]],
                            on=["night_id", PID], how="inner", validate="one_to_one",
                            suffixes=("_feature", "_prediction"))
    if len(joined) != len(features):
        raise SystemExit("Global HGB predictions do not cover every test feature night.")
    if not (joined["label_feature"] == joined["label_prediction"]).all():
        raise SystemExit("Prediction labels do not match the frozen test labels.")

    baseline = joined[[PID, "age_group", "label_feature", "prob"]].rename(
        columns={"label_feature": "label"})
    at_night = joined[[PID, "age_group_at_night", "label_feature", "prob"]].rename(
        columns={"age_group_at_night": "age_group", "label_feature": "label"})

    child_ids = set(at_night.loc[at_night.age_group == "child 0-13", PID])
    adult_ids = set(at_night.loc[at_night.age_group == "adult 18+", PID])
    if child_ids & adult_ids:
        raise SystemExit("A subject appears in both child and adult age-at-night groups; "
                         "the independent-group bootstrap would be invalid.")

    baseline_gap = primary_age_gap(baseline, args.n_boot, args.seed)
    at_night_gap = primary_age_gap(at_night, args.n_boot, args.seed)
    recorded = pd.read_csv(args.primary_result).iloc[0]
    for key, column in [("children_minus_adults", "children_minus_adults"),
                        ("ci_low", "ci_low"), ("ci_high", "ci_high")]:
        if not np.isclose(float(baseline_gap[key]), float(recorded[column]), atol=0.00002):
            raise SystemExit(f"Baseline-age check differs from saved primary result for {key}: "
                             f"{baseline_gap[key]} vs {recorded[column]}.")

    args.out.mkdir(parents=True, exist_ok=True)
    comparisons = []
    for name, result in [("age_at_first_reading", baseline),
                         ("age_at_night", at_night)]:
        gap = baseline_gap if name == "age_at_first_reading" else at_night_gap
        comparisons.append({"age_definition": name, **gap})
        groups = result.groupby("age_group", sort=True)
        for age, group in groups:
            comparisons[-1].setdefault("group_counts", {})[age] = {
                "n_subjects": int(group[PID].nunique()),
                "n_nights": int(len(group)),
                "n_positive": int(group.label.sum()),
            }

    pd.DataFrame(comparisons).to_json(args.out / "age_definition_comparison.json",
                                      orient="records", indent=2)
    features[["night_id", PID, "night_date", "age_group", "age_value_at_night",
              "age_group_at_night"]].to_csv(args.out / "night_age_mapping.csv", index=False)
    pd.crosstab(features["age_group"], features["age_group_at_night"]).to_csv(
        args.out / "age_group_transition_counts.csv")
    print(f"Mapped {len(lookup):,} of {len(features):,} test nights; "
          f"{int((features.age_group != features.age_group_at_night).sum()):,} changed age group.")
    print(pd.crosstab(features["age_group"], features["age_group_at_night"]).to_string())
    print("Baseline-age child-minus-adult:", baseline_gap["children_minus_adults"],
          (baseline_gap["ci_low"], baseline_gap["ci_high"]))
    print("Age-at-night child-minus-adult:", at_night_gap["children_minus_adults"],
          (at_night_gap["ci_low"], at_night_gap["ci_high"]))
    print(f"Saved outputs in {args.out}")


if __name__ == "__main__":
    main()
