"""Prompt 6 - baseline models: logistic regression and HistGradientBoosting.

What it does
  1. Loads results/features/features_<tag>_{train,val,test}.csv (from features.py).
  2. For each model type x scope:
       scope "global"      : train on ALL ages
       scope "<age group>" : train only on that age group (age-specific)
     tries every setting in a small grid, fits on TRAIN, scores on VALIDATION,
     keeps the best (by --tune-metric, default AUROC). The test set is not
     touched during tuning. The final model is the one fitted on train only.
  3. Picks ONE decision threshold per model from the VALIDATION set: the highest
     threshold that still catches >= --target-sens (default 80%) of validation
     positives. That threshold is then fixed and applied to test.
  4. On TEST, per age group (+ "All"): AUROC, AUPRC, sensitivity, specificity,
     PPV, false alarms per 10 nights (= per patient-night x 10), Brier score and
     expected calibration error, with 95% CIs (calibration intercept/slope:
     point estimate only) from a subject-level
     bootstrap (resample SUBJECTS with replacement, keep all their nights;
     1,000 resamples, fixed seed).
  5. Paired comparison: age-specific minus global AUROC/AUPRC on the SAME
     bootstrap resamples, with 95% CI.
  6. Between-group comparison: AUROC of each age group minus the ADULT reference
     group (the youngest group whose lower age bound is >= 18)
     for each model (independent bootstraps), with 95% CI.

Missing values: logistic regression fills NaN with TRAIN medians (fitted inside
the pipeline, so val/test never influence them). HistGradientBoosting handles
NaN natively.

Usage
  python src/baselines.py                      # primary tag from diadata_io.PRIMARY_TAG
Outputs
  results/baseline_results<suffix>.csv          main table (one row per model x scope x eval group)
  results/baseline_paired_diff<suffix>.csv      age-specific - global, with CIs
  results/baseline_group_diff<suffix>.csv       group - adult reference group, with CIs
  results/baseline_tuning<suffix>.csv           every grid setting and its val score
  results/baseline_val_scores<suffix>.csv       best val score per model x scope
  results/baseline_test_predictions<suffix>.csv per-night test probabilities (for Prompt 8)
"""
from __future__ import annotations

import argparse
import itertools
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import diadata_io as io
from features import FEATURE_COLS

PID = "PtID"

LR_GRID = {"C": [0.01, 0.1, 1.0, 10.0]}
HGB_GRID = {"learning_rate": [0.03, 0.1], "max_leaf_nodes": [15, 31],
            "min_samples_leaf": [20, 50], "l2_regularization": [0.0, 1.0]}


def build(model: str, params: dict, seed: int):
    if model == "logreg":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(C=params["C"], class_weight="balanced",
                                                max_iter=5000))
    return HistGradientBoostingClassifier(
        max_iter=300, early_stopping=False, class_weight="balanced",
        random_state=seed, **params)


def grid(model: str):
    g = LR_GRID if model == "logreg" else HGB_GRID
    keys = list(g)
    for vals in itertools.product(*(g[k] for k in keys)):
        yield dict(zip(keys, vals))


def score(y, p, metric):
    if len(np.unique(y)) < 2:
        return np.nan
    return roc_auc_score(y, p) if metric == "auroc" else average_precision_score(y, p)


def threshold_for_sensitivity(y, p, target):
    """Highest threshold whose sensitivity on (y, p) is >= target."""
    pos = np.sort(p[y == 1])[::-1]
    if len(pos) == 0:
        return np.nan
    k = int(np.ceil(target * len(pos))) - 1          # need k+1 positives caught
    return float(pos[min(k, len(pos) - 1)])


def metrics(y, p, thr) -> dict:
    pred = p >= thr
    tp = int((pred & (y == 1)).sum()); fn = int((~pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum()); tn = int((~pred & (y == 0)).sum())
    both = len(np.unique(y)) == 2
    with np.errstate(invalid="ignore", divide="ignore"):
        return {
            "auroc": roc_auc_score(y, p) if both else np.nan,
            "auprc": average_precision_score(y, p) if both else np.nan,
            "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
            "specificity": tn / (tn + fp) if tn + fp else np.nan,
            "ppv": tp / (tp + fp) if tp + fp else np.nan,
            "false_alarms_per_10_nights": 10 * fp / len(y) if len(y) else np.nan,
            "brier": float(np.mean((p - y) ** 2)) if len(y) else np.nan,
            "ece": expected_calibration_error(y, p),
        }


METRICS = ["auroc", "auprc", "sensitivity", "specificity", "ppv",
           "false_alarms_per_10_nights", "brier", "ece"]


def expected_calibration_error(y, p, n_bins: int = 10) -> float:
    """Mean |observed rate - mean predicted prob| over 10 equal-width bins,
    weighted by bin size. NOTE: class_weight='balanced' pushes probabilities up,
    so raw models will look poorly calibrated - report it honestly."""
    if len(y) == 0:
        return np.nan
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    err = 0.0
    for b in np.unique(bins):
        m = bins == b
        err += m.sum() * abs(y[m].mean() - p[m].mean())
    return float(err / len(y))


def calibration_intercept_slope(y, p):
    """Logistic recalibration: logit(P(y=1)) = a + b*logit(p). Ideal a=0, b=1."""
    if len(np.unique(y)) < 2:
        return np.nan, np.nan
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    m = LogisticRegression(C=1e6, max_iter=1000).fit(z.reshape(-1, 1), y)
    return float(m.intercept_[0]), float(m.coef_[0, 0])


class SubjectBootstrap:
    """Resample subjects with replacement; reuse the same draws for every model
    so paired differences are valid."""

    def __init__(self, subjects: np.ndarray, n_boot: int, seed: int):
        self.uniq, inv = np.unique(subjects, return_inverse=True)
        self.rows = [np.flatnonzero(inv == i) for i in range(len(self.uniq))]
        rng = np.random.default_rng(seed)
        self.draws = rng.integers(0, len(self.uniq), size=(n_boot, len(self.uniq)))

    def indices(self):
        for d in self.draws:
            yield np.concatenate([self.rows[i] for i in d])


def ci(values) -> tuple[float, float]:
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    if len(v) < 20:
        return np.nan, np.nan
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default=io.PRIMARY_TAG)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--tune-metric", choices=["auroc", "auprc"], default="auroc")
    ap.add_argument("--target-sens", type=float, default=0.80)
    ap.add_argument("--exclude-low-in-progress", action="store_true")
    ap.add_argument("--exclude-source", default=None,
                    help="drop source dataset(s) inside DiaData, comma-separated (Prompt 10)")
    ap.add_argument("--exclude-label", default=None,
                    help="short name used in output file names instead of the source list")
    ap.add_argument("--min-pos", type=int, default=10,
                    help="skip an age-specific model if train or val has fewer positives")
    ap.add_argument("--results", type=Path, default=Path("results"))
    args = ap.parse_args()
    warnings.filterwarnings("ignore", category=UserWarning)

    fdir = args.results / "features"
    data = {s: pd.read_csv(fdir / f"features_{args.tag}_{s}.csv", dtype={PID: str})
            for s in ("train", "val", "test")}
    if args.exclude_low_in_progress:
        data = {s: d[d["low_in_progress_at_start"] == 0] for s, d in data.items()}
    if args.exclude_source:
        drop = [x.strip() for x in args.exclude_source.split(",") if x.strip()]
        data = {s: d[~d["source_dataset"].isin(drop)] for s, d in data.items()}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        assert not set(data[a][PID]) & set(data[b][PID]), f"subject leak {a}/{b}"

    groups = sorted(data["test"]["age_group"].unique())
    suffix = (f"_{args.tag}" + ("_noLIP" if args.exclude_low_in_progress else "")
              + (f"_no-{args.exclude_label or args.exclude_source.replace(',', '+')}"
                 if args.exclude_source else ""))

    # ---------- train + tune ----------
    tuning_rows, preds, thresholds, skipped = [], [], {}, []
    for model in ("logreg", "hgb"):
        for scope in ["global"] + groups:
            sel = {s: (d if scope == "global" else d[d["age_group"] == scope])
                   for s, d in data.items()}
            ytr, yva = sel["train"]["label"].to_numpy(), sel["val"]["label"].to_numpy()
            if ytr.sum() < args.min_pos or yva.sum() < args.min_pos:
                skipped.append(f"{model}/{scope}: train pos={ytr.sum()}, val pos={yva.sum()}")
                continue
            Xtr, Xva = sel["train"][FEATURE_COLS], sel["val"][FEATURE_COLS]
            best, best_s = None, -np.inf
            for params in grid(model):
                m = build(model, params, args.seed).fit(Xtr, ytr)
                s_val = score(yva, m.predict_proba(Xva)[:, 1], args.tune_metric)
                tuning_rows.append({"model": model, "scope": scope, **params,
                                    f"val_{args.tune_metric}": s_val})
                if s_val > best_s:
                    best, best_s = (params, m), s_val
            params, m = best
            thr = threshold_for_sensitivity(yva, m.predict_proba(Xva)[:, 1], args.target_sens)
            thresholds[(model, scope)] = thr
            print(f"{model:6s} {scope:>18s}: best {params}  val {args.tune_metric}={best_s:.3f}  "
                  f"threshold={thr:.3f}")
            te = sel["test"]
            preds.append(pd.DataFrame({
                "night_id": te["night_id"].to_numpy(), PID: te[PID].to_numpy(),
                "age_group": te["age_group"].to_numpy(), "label": te["label"].to_numpy(),
                "model": model, "scope": scope,
                "prob": m.predict_proba(te[FEATURE_COLS])[:, 1], "threshold": thr}))
    best_val = (pd.DataFrame(tuning_rows).groupby(["model", "scope"])[f"val_{args.tune_metric}"]
                .max().reset_index())
    best_val.to_csv(args.results / f"baseline_val_scores{suffix}.csv", index=False)
    pd.DataFrame(tuning_rows).to_csv(args.results / f"baseline_tuning{suffix}.csv", index=False)
    P = pd.concat(preds, ignore_index=True)
    P["pred"] = (P["prob"] >= P["threshold"]).astype(int)
    P.to_csv(args.results / f"baseline_test_predictions{suffix}.csv", index=False)

    # ---------- evaluate with subject bootstrap ----------
    test = data["test"]
    results, paired, boot_auroc = [], [], {}
    # One INDEPENDENT bootstrap stream per evaluation group (deterministic seed per
    # group), so between-group differences use independent resamples. Within a group
    # all models share the same draws, so paired model comparisons stay paired.
    group_seed = {g: args.seed + 1000 * (i + 1) for i, g in enumerate(["All"] + groups)}
    for g in ["All"] + groups:
        sub = test if g == "All" else test[test["age_group"] == g]
        bs = SubjectBootstrap(sub[PID].to_numpy(), args.n_boot, group_seed[g])
        idx_list = list(bs.indices())
        night_order = sub["night_id"].to_numpy()
        per_model = {}
        for (model, scope), thr in thresholds.items():
            if g == "All" and scope != "global":
                continue
            if g != "All" and scope not in ("global", g):
                continue
            pm = P[(P.model == model) & (P.scope == scope)].set_index("night_id")
            if not pm.index.isin(night_order).any():
                continue
            pm = pm.loc[night_order]
            y, p = pm["label"].to_numpy(), pm["prob"].to_numpy()
            point = metrics(y, p, thr)
            boots = [metrics(y[ix], p[ix], thr) for ix in idx_list]
            row = {"model": model, "trained_on": scope if scope == "global" else "age-specific",
                   "eval_group": g, "n_subjects": len(bs.uniq), "n_nights": len(y),
                   "n_positive": int(y.sum()), "prevalence": round(y.mean(), 4),
                   "threshold": round(thr, 4),
                   "ci_warning": ("fewer than 10 test subjects - CI unreliable"
                                  if len(bs.uniq) < 10 else "")}
            for k in METRICS:
                lo, hi = ci([b[k] for b in boots])
                row[k], row[f"{k}_ci_low"], row[f"{k}_ci_high"] = point[k], lo, hi
            row["calibration_intercept"], row["calibration_slope"] = calibration_intercept_slope(y, p)
            results.append(row)
            per_model[(model, row["trained_on"])] = boots
            if row["trained_on"] == "global":
                boot_auroc[(model, g)] = (point["auroc"], [b["auroc"] for b in boots])
        for model in ("logreg", "hgb"):
            a, b = per_model.get((model, "age-specific")), per_model.get((model, "global"))
            if a is None or b is None:
                continue
            for k in ("auroc", "auprc"):
                d = [x[k] - z[k] for x, z in zip(a, b)]
                pa = next(r[k] for r in results if r["model"] == model and r["eval_group"] == g
                          and r["trained_on"] == "age-specific")
                pb = next(r[k] for r in results if r["model"] == model and r["eval_group"] == g
                          and r["trained_on"] == "global")
                lo, hi = ci(d)
                paired.append({"model": model, "eval_group": g, "metric": k,
                               "age_specific_minus_global": pa - pb, "ci_low": lo, "ci_high": hi})

    # between-group: each group minus the adult reference group (global model).
    # PRIMARY comparison (analysis_plan.md): youngest group (children) minus adults.
    gdiff = []
    if groups:
        def _lower(gname):
            m = re.search(r"\d+", str(gname))
            return float(m.group()) if m else np.nan
        adults = sorted([g for g in groups if _lower(g) >= 18], key=_lower)
        ref = adults[0] if adults else sorted(groups, key=_lower)[-1]
        for model in ("logreg", "hgb"):
            if (model, ref) not in boot_auroc:
                continue
            r_pt, r_b = boot_auroc[(model, ref)]
            for g in [x for x in groups if x != ref]:
                if (model, g) not in boot_auroc:
                    continue
                g_pt, g_b = boot_auroc[(model, g)]
                lo, hi = ci(np.array(g_b) - np.array(r_b))   # independent subjects
                gdiff.append({"model": model, "comparison": f"{g} minus {ref}",
                              "auroc_diff": g_pt - r_pt, "ci_low": lo, "ci_high": hi})

    res = pd.DataFrame(results)
    out = args.results / ("baseline_results.csv" if suffix == f"_{io.PRIMARY_TAG}"
                          else f"baseline_results{suffix}.csv")
    res.round(4).to_csv(out, index=False)
    pd.DataFrame(paired).round(4).to_csv(args.results / f"baseline_paired_diff{suffix}.csv", index=False)
    pd.DataFrame(gdiff).round(4).to_csv(args.results / f"baseline_group_diff{suffix}.csv", index=False)

    if skipped:
        print("\nSkipped (too few positives):\n  " + "\n  ".join(skipped))
    show = ["model", "trained_on", "eval_group", "n_nights", "prevalence", "auroc",
            "auroc_ci_low", "auroc_ci_high", "auprc", "sensitivity", "specificity",
            "false_alarms_per_10_nights"]
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print("\n", res[show].round(3).to_string(index=False))
        if gdiff:
            print(f"\nGlobal-model AUROC difference vs adult reference group ({ref}):\n",
                  pd.DataFrame(gdiff).round(3).to_string(index=False))
    print(f"\nSaved {out.name} and companion files in {args.results}/")
    print("Reminder: AUPRC depends on prevalence - compare it to the 'prevalence' column, "
          "which is the AUPRC of a random guess.")


if __name__ == "__main__":
    main()
