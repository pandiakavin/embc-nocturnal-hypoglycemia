"""Prompt 5 - patient-level split + evening features.

STEP A - split SUBJECTS (never nights) into train / val / test = 70 / 15 / 15,
stratified by age group, with a fixed seed. The split is built ONCE from
results/subject_summary.csv (made by explore_data.py) and saved to
results/splits_seed<seed>.csv. Every later run (including the Prompt 10
robustness runs with other night definitions) REUSES that file, so a subject
is always in the same split. Subjects with unknown age are left out, because
they cannot be compared by age.

How the stratified split works (simple and transparent): within each age
group, sort the subject IDs, shuffle them with the seeded random generator,
then give the first round(70%) to train, the next round(15%) to val and the
rest to test.

STEP B - features from each night's evening window (default 18:00-24:00,
72 readings, NaN = missing). Nights whose evening has <80% of readings are
dropped (option --min-evening-coverage).

  last_glucose     last reading in the final 15 min of the evening (NaN if none)
  mean, sd, cv     over available readings; cv = sd / mean * 100 (%)
  min              lowest evening reading
  roc_30, roc_60   rate of change in mg/dL per minute = least-squares slope of
                   the readings in the last 30 / 60 min (7 / 13 points). Needs at
                   least half the points present, else NaN. A slope uses every
                   point, so one noisy or missing reading matters less than in
                   "last minus first".
  pct_below_70     % of available evening readings < 70 mg/dL
  pct_above_180    % of available evening readings > 180 mg/dL
  prior24h_low_episodes  from nights.py: episodes in the 24 h before the night
                   starts; an episode = a run meeting the label rule (primary: >=4
                   consecutive readings <70) and ends only after >=3 OBSERVED
                   non-low readings (15 min); closer runs are merged. NOTE this
                   window includes the evening window.

Missing feature values stay NaN here. HistGradientBoosting handles NaN
itself; for logistic regression (Prompt 6) fill NaN with TRAIN-set medians
only, never with statistics from val/test.

Usage:
  python src/features.py                      # primary settings (diadata_io.PRIMARY)
  python src/features.py --include-ambiguous  # sensitivity: gap-hidden lows counted negative
  python src/features.py --keep-ongoing       # sensitivity: include events already under way at 00:00
Outputs:
  results/splits_seed42.csv                   PtID, age_group, split
  results/features/features_<tag>_train.csv   (and _val, _test)
  results/features/split_summary_<tag>.csv    subjects / nights / % positive
"""
from __future__ import annotations
# NOTE: if --include-ambiguous / --keep-ongoing are used, the output files are
# overwritten for that tag; run baselines right after, or copy results first.

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import diadata_io as io

FEATURE_COLS = ["last_glucose", "mean", "sd", "cv", "min", "roc_30", "roc_60",
                "pct_below_70", "pct_above_180", "prior24h_low_episodes"]
META_COLS = ["night_id", io.PID, "age_group", "source_dataset", "night_date",
             "split", "label", "evening_coverage", "night_coverage",
             "low_in_progress_at_start", "onset_min_after_start", "night_nadir"]
SPLITS = ("train", "val", "test")


# ---------------- STEP A: split ----------------
def make_split(subjects: pd.DataFrame, seed: int, fracs=(0.70, 0.15)) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = []
    for group in sorted(subjects["age_group"].unique()):
        ids = np.array(sorted(subjects.loc[subjects["age_group"] == group, io.PID]))
        rng.shuffle(ids)
        n = len(ids)
        n_train = int(round(fracs[0] * n))
        n_val = int(round(fracs[1] * n))
        labels = (["train"] * n_train + ["val"] * n_val + ["test"] * (n - n_train - n_val))
        out.append(pd.DataFrame({io.PID: ids, "age_group": group, "split": labels}))
    return pd.concat(out, ignore_index=True)


def verify_split(split: pd.DataFrame) -> None:
    """Hard checks - the script stops if any fails."""
    assert split[io.PID].is_unique, "a subject is listed twice in the split file"
    sets = {s: set(split.loc[split["split"] == s, io.PID]) for s in SPLITS}
    for a in SPLITS:
        for b in SPLITS:
            if a < b:
                overlap = sets[a] & sets[b]
                assert not overlap, f"subjects in both {a} and {b}: {sorted(overlap)[:5]}"
    assert set(split["split"]) <= set(SPLITS)


# ---------------- STEP B: features ----------------
def slope_last(ev: np.ndarray, n_points: int) -> np.ndarray:
    """Least-squares slope (mg/dL per minute) over the last n_points readings."""
    y = ev[:, -n_points:].astype("float64")
    t = np.arange(n_points) * io.SLOT_MIN               # minutes
    ok = ~np.isnan(y)
    n = ok.sum(axis=1)
    tt = np.where(ok, t, 0.0)
    yy = np.where(ok, y, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        t_mean = tt.sum(1) / n
        y_mean = yy.sum(1) / n
        cov = (np.where(ok, (t - t_mean[:, None]) * (y - y_mean[:, None]), 0)).sum(1)
        var = (np.where(ok, (t - t_mean[:, None]) ** 2, 0)).sum(1)
        s = cov / var
    s[n < (n_points + 1) // 2] = np.nan                 # need >= half the points
    return s


def last_reading(ev: np.ndarray, within: int = 3) -> np.ndarray:
    tail = ev[:, -within:]
    out = np.full(len(ev), np.nan)
    for k in range(within):                             # walk back from the end
        col = tail[:, within - 1 - k]
        fill = np.isnan(out) & ~np.isnan(col)
        out[fill] = col[fill]
    return out


def evening_features(ev: np.ndarray) -> pd.DataFrame:
    ok = ~np.isnan(ev)
    n_ok = ok.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN rows -> NaN
        mean = np.nanmean(ev, axis=1)
        sd = np.nanstd(ev, axis=1, ddof=1)
        mn = np.nanmin(ev, axis=1)
        below = 100 * (np.where(ok, ev < 70, False).sum(1)) / n_ok
        above = 100 * (np.where(ok, ev > 180, False).sum(1)) / n_ok
    return pd.DataFrame({
        "last_glucose": last_reading(ev),
        "mean": mean,
        "sd": sd,
        "cv": 100 * sd / mean,
        "min": mn,
        "roc_30": slope_last(ev, 30 // io.SLOT_MIN + 1),
        "roc_60": slope_last(ev, 60 // io.SLOT_MIN + 1),
        "pct_below_70": below,
        "pct_above_180": above,
    })


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default=io.PRIMARY_TAG, help="matches nights.py output names")
    ap.add_argument("--keep-ongoing", action="store_true",
                    help="keep nights whose qualifying low event began before night start and "
                         "overlaps it "
                         "(default: drop them, so the target is NEW-ONSET overnight lows)")
    ap.add_argument("--variant-suffix", default="",
                    help="appended to the tag for ALL outputs (e.g. _amb), so sensitivity "
                         "runs never overwrite the primary files; the windows file is "
                         "copied under the new name for later scripts")
    ap.add_argument("--include-ambiguous", action="store_true",
                    help="sensitivity: keep negative nights where a data gap may have hidden a "
                         "low, counting them as negative (default: drop them = 'unknown')")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-evening-coverage", type=float, default=0.80)
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
    args = ap.parse_args()

    # ---- STEP A ----
    split_path = args.results / f"splits_seed{args.seed}.csv"
    if split_path.exists():
        split = pd.read_csv(split_path, dtype={io.PID: str})
        print(f"Reusing existing split {split_path} (delete it only if you mean to re-split)")
    else:
        subj = pd.read_csv(args.results / "subject_summary.csv", dtype={io.PID: str})
        subj = subj[subj["age_group"] != io.UNKNOWN]
        split = make_split(subj[[io.PID, "age_group"]], args.seed)
        split.to_csv(split_path, index=False)
        print(f"Created {split_path}")
    verify_split(split)
    counts = split.groupby(["age_group", "split"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=list(SPLITS), fill_value=0)
    print("Subjects per split:\n", counts.to_string())
    if (counts == 0).any().any():
        print("WARNING: some age group has NO subjects in a split - per-group test "
              "results will be missing for it. Consider merging tiny groups.")
    print("Check passed: no subject appears in more than one split.")

    # ---- STEP B ----
    nights = pd.read_csv(args.results / f"nights_{args.tag}.csv", dtype={io.PID: str})
    out_tag = args.tag + args.variant_suffix
    win = np.load(args.processed / f"windows_{args.tag}.npz")
    assert (win["night_id"] == nights["night_id"].to_numpy()).all(), "CSV and arrays misaligned"
    ev = win["evening"]

    feats = evening_features(ev)
    # prior24h_low_episodes is already a column of the nights CSV
    df = pd.concat([nights.reset_index(drop=True), feats], axis=1)
    df = df.drop(columns="age_group").merge(split, on=io.PID, how="inner")

    n_before = len(nights)
    n_no_split = n_before - nights[io.PID].isin(split[io.PID]).sum()
    # drop reasons, applied in this order; each night gets the FIRST reason that applies
    reason = pd.Series("kept", index=df.index)
    reason[df["evening_coverage"] < args.min_evening_coverage] = "evening_coverage"
    if not args.keep_ongoing:
        reason[(reason == "kept") & (df["low_in_progress_at_start"] == 1)] = "low_ongoing_at_start"
    if not args.include_ambiguous and "label_ambiguous_gap" in df:
        reason[(reason == "kept") & (df["label_ambiguous_gap"] == 1)] = "ambiguous_gap"
    drops = pd.crosstab(df["age_group"], reason, margins=True, margins_name="All")
    print(f"Nights: {n_before} kept by nights.py; {n_no_split} have no split (e.g. unknown age).")
    print("Nights kept / dropped by reason and age group:\n", drops.to_string())
    df = df[reason == "kept"]

    # night-level re-check: every night's subject belongs to exactly one split
    assert df.groupby(io.PID)["split"].nunique().max() == 1, "subject's nights in >1 split!"

    out_dir = args.results / "features"
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = [c for c in META_COLS if c in df.columns] + FEATURE_COLS
    for s in SPLITS:
        part = df.loc[df["split"] == s, cols].sort_values("night_id")
        part.to_csv(out_dir / f"features_{out_tag}_{s}.csv", index=False)

    summ = (df.groupby(["split", "age_group"])
              .agg(n_subjects=(io.PID, "nunique"), n_nights=("label", "size"),
                   n_positive=("label", "sum"), pct_positive=("label", "mean"))
              .reset_index())
    summ["pct_positive"] = (100 * summ["pct_positive"]).round(2)
    summ.to_csv(out_dir / f"split_summary_{out_tag}.csv", index=False)
    drops.to_csv(out_dir / f"drop_reasons_{out_tag}.csv")
    if args.variant_suffix:
        import shutil
        shutil.copyfile(args.processed / f"windows_{args.tag}.npz",
                        args.processed / f"windows_{out_tag}.npz")
    with pd.option_context("display.width", 200):
        print("\n", summ.to_string(index=False))
        print("\nMissing feature values (% of nights):")
        print((100 * df[FEATURE_COLS].isna().mean()).round(2).to_string())
    print(f"\nSaved features to {out_dir}/")


if __name__ == "__main__":
    main()
