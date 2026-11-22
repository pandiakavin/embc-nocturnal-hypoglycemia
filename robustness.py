"""Prompt 10 - robustness: do the main conclusions survive other choices?

Variants (all reuse the SAME subject split, results/splits_seed42.csv, and the
SAME model type as the main analysis). Primary = diadata_io.PRIMARY:
  main            00:00-06:00, <70 mg/dL for >=4 readings, new-onset, gap-ambiguous excluded
  night_22_07     night 22:00-07:00 (evening window moves to 16:00-22:00)
  thr_lt54        <54 mg/dL (Level 2)
  old_rule        <=70 mg/dL for >=3 readings (closer to Cinar & Maleshkova)
  interp_lt30     gaps of <=4 missing readings interpolated before labelling
  ambiguous_neg   gap-ambiguous nights kept and counted negative
  keep_ongoing    lows already under way at 00:00 kept
  drop_largest    main, without the largest source dataset
  exclude_weak_hourly_profiles main, without sources with flat/sparse hourly profiles
  exclude_reported_10_15m main, without four sources with reported 10/15-min sampling

Conclusions checked in every variant (sig = 95% CI excludes 0):
  C1  AUROC children minus adults (global model) - the PRIMARY comparison
  C2  AUROC age-specific minus global, for children (paired bootstrap)
  C3  miss rate children minus adults (from error_analysis)
Verdicts: holds / holds (no clear effect) / weaker / new effect / flips.

Usage
  python src/robustness.py --main data/raw/MDB_5min_sampling_raw.csv \
      --demographics data/raw/SDBI_5min_sampling_raw.csv
Output
  results/robustness_summary.csv   (one row per variant)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import diadata_io as io
from baselines import SubjectBootstrap, ci
from error_analysis import choose_model, lower_age

SRC = Path(__file__).resolve().parent
PID = "PtID"
# fixed in paper/analysis_plan.md BEFORE this variant was run
WEAK_HOURLY_PROFILE = "RT-CGM,D1NAMO,DDATSHR,HUPA-UCM,ShanghaiT1D,T1GDUJA"
REPORTED_10_15_MIN = "RT-CGM,HUPA-UCM,DDATSHR,ShanghaiT1D"


def run(script, *a):
    cmd = [sys.executable, str(SRC / script), *map(str, a)]
    print("  $", " ".join(cmd[1:]))
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)


def cohort(g):
    lo = lower_age(g)
    return "children" if lo < 14 else "adults" if lo >= 18 else "other"


def auroc_gap(pred: pd.DataFrame, model: str, n_boot: int, seed: int):
    """C1: AUROC(children) - AUROC(adults) for the global model."""
    pred_model = "cnn_5seed_avg" if model == "cnn" else model
    p = pred[(pred.model == pred_model) & (pred.scope == "global")].copy()
    p["cohort"] = p.age_group.map(cohort)
    out = {}
    # Match the primary baseline/error-analysis streams exactly (sorted groups:
    # adult=seed+2000, child=seed+3000). This keeps the robustness main row's
    # primary CI identical to the prespecified result while preserving independent
    # resampling between age groups and deterministic streams in every variant.
    seeds = {"children": seed + 3000, "adults": seed + 2000}
    for i, c in enumerate(("children", "adults")):
        d = p[p.cohort == c]
        bs = SubjectBootstrap(d[PID].to_numpy(), n_boot, seeds[c])
        y, s = d.label.to_numpy(), d.prob.to_numpy()
        b = [roc_auc_score(y[i], s[i]) if len(np.unique(y[i])) == 2 else np.nan
             for i in bs.indices()]
        out[c] = (roc_auc_score(y, s), np.array(b))
    d_pt = out["children"][0] - out["adults"][0]
    lo, hi = ci(out["children"][1] - out["adults"][1])
    return out["children"][0], out["adults"][0], d_pt, lo, hi


def verdict(main, this):
    """Compare (point, lo, hi) of a variant with the main analysis.
    sig = the 95% CI excludes 0."""
    if any(pd.isna(v) for v in (*main, *this)):
        return "n/a"
    sig = lambda t: t[1] > 0 or t[2] < 0
    sm, sv = sig(main), sig(this)
    if not sm and not sv:
        return "holds (no clear effect in either)"
    if sm and sv:
        return "holds" if np.sign(main[0]) == np.sign(this[0]) else "flips"
    if sm and not sv:
        return "weaker (CI now includes 0)"
    return "new effect (CI excludes 0 only here)"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--main", type=Path, required=True)
    ap.add_argument("--demographics", type=Path, default=None)
    ap.add_argument("--age-scheme", default="kids_teens_adults")
    ap.add_argument("--model", choices=["auto", "logreg", "hgb", "cnn"], default="auto",
                    help="auto = global model selected from validation AUROC in the main analysis")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true", help="rerun steps even if outputs exist")
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
    args = ap.parse_args()
    R, PR = args.results, args.processed
    common = ["--main", args.main, "--age-scheme", args.age_scheme, "--results", R,
              "--processed", PR] + (["--demographics", args.demographics] if args.demographics else [])

    by_db = pd.read_csv(R / "data_summary_by_database.csv")
    by_db = by_db[by_db.source_dataset != "All"]
    largest = str(by_db.sort_values("n_subjects", ascending=False).iloc[0].source_dataset)

    P = io.PRIMARY_TAG
    # (name, nights tag, nights.py args, features.py args, features variant suffix, dropped source)
    variants = [
        ("main", P, [], [], "", None, None),
        ("night_22_07", "n22-07_lt70_r4", ["--night-start", 22, "--night-end", 7], [], "", None, None),
        ("thr_lt54", "n00-06_lt54_r4", ["--threshold", 54], [], "", None, None),
        ("old_rule", "n00-06_le70_r3", ["--op", "le", "--min-run", 3], [], "", None, None),
        ("interp_lt30", P + "_i4", ["--interp-max-missing", 4], [], "", None, None),
        ("ambiguous_neg", P, [], ["--include-ambiguous"], "_amb", None, None),
        ("keep_ongoing", P, [], ["--keep-ongoing"], "_ongoing", None, None),
        ("drop_largest", P, [], [], "", largest, "largest"),
        ("exclude_weak_hourly_profiles", P, [], [], "", WEAK_HOURLY_PROFILE, "weak-hours"),
        ("exclude_reported_10_15m", P, [], [], "", REPORTED_10_15_MIN, "coarse-sampling"),
    ]

    rows, main_vals = [], None
    main_pred_path = None
    model = args.model
    if model == "auto":
        ch = R / "error_analysis" / P / "chosen.json"
        model = json.loads(ch.read_text())["model"] if ch.exists() else choose_model(R, f"_{P}")[0]
    print(f"Selected model used for every variant: {model}")
    for name, ntag, night_args, feat_args, vsuf, drop, drop_label in variants:
        tag = ntag + vsuf
        print(f"\n=== {name} ({tag}{', without ' + drop if drop else ''}) ===")
        lab = drop_label or ""
        extra = f"_no-{lab}" if drop else ""
        suffix = f"_{tag}{extra}"
        if args.force or not (R / f"nights_{ntag}.csv").exists():
            run("nights.py", *common, *night_args)
        if args.force or not (R / "features" / f"features_{tag}_test.csv").exists():
            run("features.py", "--tag", ntag, "--variant-suffix", vsuf, *feat_args,
                "--seed", args.seed, "--results", R, "--processed", PR)
        baseline_pred_path = R / f"baseline_test_predictions{suffix}.csv"
        if args.force or not baseline_pred_path.exists():
            run("baselines.py", "--tag", tag, "--seed", args.seed, "--n-boot", args.n_boot,
                "--results", R, *(["--exclude-source", drop, "--exclude-label", lab] if drop else []))
        if model == "cnn":
            pred_path = R / f"cnn_test_predictions{suffix}.csv"
            if args.force or not pred_path.exists():
                run("cnn.py", "--tag", tag, "--seed", args.seed, "--n-boot", args.n_boot,
                    "--results", R, "--processed", PR,
                    *(["--exclude-source", drop, "--exclude-label", lab] if drop else []))
        else:
            pred_path = baseline_pred_path
        if name == "main":
            main_pred_path = pred_path
        ea_dir = R / "error_analysis" / f"{tag}{extra}"
        if args.force or not (ea_dir / "miss_rates.csv").exists() or name == "main":
            run("error_analysis.py", "--tag", tag, "--suffix-extra", extra, "--model", model,
                "--seed", args.seed, "--n-boot", args.n_boot, "--results", R)

        pred = pd.read_csv(pred_path, dtype={PID: str})
        a_ch, a_ad, c1, c1lo, c1hi = auroc_gap(pred, model, args.n_boot, args.seed)
        try:
            pdiff = pd.read_csv(R / f"baseline_paired_diff{suffix}.csv")
        except pd.errors.EmptyDataError:      # no age-specific models (too few positives)
            pdiff = pd.DataFrame(columns=["model", "eval_group", "metric",
                                          "age_specific_minus_global", "ci_low", "ci_high"])
        ch_groups = [g for g in pdiff.eval_group.unique() if cohort(g) == "children"]
        c2 = pdiff[(pdiff.model == model) & (pdiff.metric == "auroc")
                   & pdiff.eval_group.isin(ch_groups)]
        c2 = (c2.iloc[0].age_specific_minus_global, c2.iloc[0].ci_low, c2.iloc[0].ci_high) \
            if len(c2) else (np.nan,) * 3
        mr = pd.read_csv(ea_dir / "miss_rates.csv").set_index("group")
        c3 = tuple(mr.loc["children minus adults", ["miss_rate", "ci_low", "ci_high"]]) \
            if "children minus adults" in mr.index else (np.nan,) * 3
        test = pd.read_csv(R / "features" / f"features_{tag}_test.csv", dtype={PID: str})
        if drop:
            test = test[~test.source_dataset.isin(drop.split(","))]
        test["cohort"] = test.age_group.map(cohort)
        prev = test.groupby("cohort").label.mean()
        vals = {"C1": (c1, c1lo, c1hi), "C2": c2, "C3": c3}
        if name == "main":
            main_vals = vals
        rows.append({
            "variant": name, "dropped_source": drop or "",
            "n_test_nights": len(test), "n_test_positive": int(test.label.sum()),
            "prevalence_children": prev.get("children", np.nan),
            "prevalence_adults": prev.get("adults", np.nan),
            "auroc_children": a_ch, "auroc_adults": a_ad,
            "C1_auroc_children_minus_adults": c1, "C1_ci_low": c1lo, "C1_ci_high": c1hi,
            "C1_vs_main": verdict(main_vals["C1"], vals["C1"]),
            "C2_children_agespecific_minus_global": c2[0], "C2_ci_low": c2[1], "C2_ci_high": c2[2],
            "C2_vs_main": verdict(main_vals["C2"], vals["C2"]),
            "C3_missrate_children_minus_adults": c3[0], "C3_ci_low": c3[1], "C3_ci_high": c3[2],
            "C3_vs_main": verdict(main_vals["C3"], vals["C3"]),
        })
    T = pd.DataFrame(rows)
    T.round(4).to_csv(R / "robustness_summary.csv", index=False)

    # Post hoc same-source check: the only test source containing both children
    # and adults. This subsets the already trained primary model's predictions.
    if main_pred_path is not None:
        primary_pred = pd.read_csv(main_pred_path, dtype={PID: str})
        selected_name = "cnn_5seed_avg" if model == "cnn" else model
        primary_pred = primary_pred[(primary_pred.model == selected_name) & (primary_pred.scope == "global")].copy()
        test_meta = pd.read_csv(R / "features" / f"features_{P}_test.csv",
                                dtype={PID: str})[["night_id", "source_dataset"]]
        primary_pred = primary_pred.merge(test_meta, on="night_id", how="inner", validate="one_to_one")
        rt = primary_pred[primary_pred.source_dataset == "RT-CGM"].copy()
        try:
            rt_ch, rt_ad, rt_delta, rt_lo, rt_hi = auroc_gap(rt, model, args.n_boot, args.seed + 7000)
            rt_rows = []
            for c in ("children", "adults"):
                sub = rt[rt.age_group.map(cohort) == c]
                rt_rows.append({"cohort": c, "n_subjects": sub[PID].nunique(),
                                "n_nights": len(sub), "n_positive_nights": int(sub.label.sum()),
                                "auroc": rt_ch if c == "children" else rt_ad})
            rt_rows.append({"cohort": "children minus adults", "n_subjects": np.nan,
                            "n_nights": np.nan, "n_positive_nights": np.nan,
                            "auroc": rt_delta, "ci_low": rt_lo, "ci_high": rt_hi})
            pd.DataFrame(rt_rows).round(5).to_csv(R / "rt_cgm_same_source.csv", index=False)
        except (ValueError, IndexError) as exc:
            (R / "rt_cgm_same_source.txt").write_text(
                f"Same-source RT-CGM age comparison unavailable: {exc}\n")
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print("\n", T.round(3).to_string(index=False))
    print(f"\nModel: {model}. Largest source dataset dropped: {largest}. "
          f"Saved {R / 'robustness_summary.csv'}")
    print("Note: the <54 variant has far fewer positive nights, so wider CIs are expected; "
          "'weaker' there may reflect less data rather than a different answer.")


if __name__ == "__main__":
    main()
