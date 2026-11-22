"""Subject-level evaluation and exploratory pre-midnight pattern analysis.

The primary model is selected using validation AUROC only. The primary age
contrast is test AUROC (children 0-13 minus adults >=18), with independent
subject-level bootstrap resampling within each age cohort. Positive test nights
are also classified as caught or missed at the validation-locked threshold.

Pattern analyses use only predictors available before 00:00. Repeated nights
are summarized within subjects before comparisons; post-midnight onset and
nadir are exported only as descriptive event characteristics. These analyses
are exploratory associations, not causal explanations.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr, wilcoxon
from sklearn.metrics import average_precision_score, roc_auc_score

from baselines import ci

PID = "PtID"
FACTORS = {
    "last_glucose": "Last evening glucose (mg/dL)",
    "mean": "Evening mean glucose (mg/dL)",
    "sd": "Evening glucose SD (mg/dL)",
    "cv": "Evening glucose CV (%)",
    "min": "Evening minimum glucose (mg/dL)",
    "roc_30": "Evening 30-min slope (mg/dL/min)",
    "roc_60": "Evening 60-min slope (mg/dL/min)",
    "pct_below_70": "Evening readings <70 mg/dL (%)",
    "pct_above_180": "Evening readings >180 mg/dL (%)",
    "prior24h_low_episodes": "Prior-24-h low episodes (count)",
    "evening_missing_pct": "Evening missing readings (%)",
}


def lower_age(label: str) -> float:
    m = re.search(r"\d+", str(label))
    return float(m.group()) if m else np.nan


def cohort(label: str) -> str:
    lo = lower_age(label)
    return "children" if lo < 14 else "adults" if lo >= 18 else "teens"


def holm(p):
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    if not ok.any():
        return out
    idx = np.argsort(p[ok])
    pv = p[ok][idx]
    m = len(pv)
    adj = np.maximum.accumulate((m - np.arange(m)) * pv)
    tmp = np.empty(m)
    tmp[idx] = np.minimum(adj, 1)
    out[ok] = tmp
    return out


def choose_model(results: Path, suffix: str) -> tuple[str, dict]:
    """Choose among global models using a single validation AUROC per model."""
    scores = {}
    bv = results / f"baseline_val_scores{suffix}.csv"
    if bv.exists():
        b = pd.read_csv(bv)
        b = b[b.scope == "global"]
        for _, r in b.iterrows():
            scores[str(r.model)] = float(r.val_auroc)
    cv = results / f"cnn_val_scores{suffix}.csv"
    if cv.exists():
        c = pd.read_csv(cv)
        if "score_type" not in c or not (c.score_type == "five_seed_average_used_for_selection").any():
            raise SystemExit("CNN validation score is per-seed best epoch; rerun cnn.py to save the five-seed ensemble AUROC before model selection.")
        scores["cnn"] = float(c.loc[c.score_type == "five_seed_average_used_for_selection", "best_val_auroc"].iloc[0])
    if not scores:
        raise SystemExit("No validation scores found - run baselines.py (and cnn.py) first.")
    return max(scores, key=scores.get), scores


def _cluster_boot_auc(d: pd.DataFrame, n_boot: int, seed: int):
    ids = d[PID].drop_duplicates().to_numpy()
    rows = {pid: d.index[d[PID] == pid].to_numpy() for pid in ids}
    y_all, p_all = d["label"], d["prob"]
    point = roc_auc_score(y_all, p_all)
    rng = np.random.default_rng(seed)
    boots = []
    for draw in rng.integers(0, len(ids), size=(n_boot, len(ids))):
        ix = np.concatenate([rows[ids[i]] for i in draw])
        yy = y_all.loc[ix].to_numpy()
        boots.append(roc_auc_score(yy, p_all.loc[ix]) if len(np.unique(yy)) == 2 else np.nan)
    return point, np.asarray(boots), len(ids)


def primary_age_gap(pred: pd.DataFrame, n_boot: int, seed: int) -> dict:
    p = pred.copy()
    p["cohort"] = p.age_group.map(cohort)
    groups = {c: p[p.cohort == c].reset_index(drop=True) for c in ("children", "adults")}
    if any(len(d) == 0 or d.label.nunique() < 2 for d in groups.values()):
        raise ValueError("Primary age comparison needs positive and negative test nights in both children and adults.")
    # Match the baseline evaluator streams (All, then sorted age labels).
    age_labels = sorted(p.age_group.unique())
    seeds = {cohort(g): seed + 1000 * (i + 2) for i, g in enumerate(age_labels)}
    ch, chb, nch = _cluster_boot_auc(groups["children"], n_boot, seeds["children"])
    ad, adb, nad = _cluster_boot_auc(groups["adults"], n_boot, seeds["adults"])
    delta = ch - ad
    lo, hi = ci(chb - adb)
    return {
        "auroc_children": ch, "n_subjects_children": nch,
        "n_nights_children": len(groups["children"]),
        "n_positive_children": int(groups["children"].label.sum()),
        "auroc_adults": ad, "n_subjects_adults": nad,
        "n_nights_adults": len(groups["adults"]),
        "n_positive_adults": int(groups["adults"].label.sum()),
        "children_minus_adults": delta, "ci_low": lo, "ci_high": hi,
        "interpretation": ("children_lower" if hi < 0 else "children_higher" if lo > 0
                           else "no_clear_difference"),
    }


def pairwise_age_differences(pred: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    """Exploratory pairwise AUROC/AUPRC differences with independent subject resampling."""
    p = pred.copy()
    p["cohort"] = p.age_group.map(cohort)
    boot = {}
    points = {}
    # Reuse the deterministic per-age streams used by the primary contrast and
    # baseline evaluator. This keeps the duplicate child-versus-adult row aligned
    # with the prespecified primary CI while still supporting every age pair.
    age_labels = sorted(p.age_group.unique())
    group_seed = {cohort(g): seed + 1000 * (i + 2) for i, g in enumerate(age_labels)}
    for c in ("children", "teens", "adults"):
        d = p[p.cohort == c].reset_index(drop=True)
        if d.empty or d.label.nunique() < 2:
            continue
        ids = d[PID].drop_duplicates().to_numpy()
        rows = {pid: np.flatnonzero(d[PID].to_numpy() == pid) for pid in ids}
        yy, pp = d.label.to_numpy(), d.prob.to_numpy()
        points[c] = {"auroc": roc_auc_score(yy, pp),
                     "auprc": average_precision_score(yy, pp),
                     "n_subjects": len(ids), "n_nights": len(d), "prevalence": yy.mean()}
        rng = np.random.default_rng(group_seed[c])
        b = {"auroc": [], "auprc": []}
        for draw in rng.integers(0, len(ids), size=(n_boot, len(ids))):
            ix = np.concatenate([rows[ids[i]] for i in draw])
            yb, pb = yy[ix], pp[ix]
            b["auroc"].append(roc_auc_score(yb, pb) if len(np.unique(yb)) == 2 else np.nan)
            b["auprc"].append(average_precision_score(yb, pb))
        boot[c] = {k: np.asarray(v) for k, v in b.items()}
    rows_out = []
    for a, b in (("children", "teens"), ("teens", "adults"), ("children", "adults")):
        if a not in points or b not in points:
            continue
        for metric in ("auroc", "auprc"):
            delta = points[a][metric] - points[b][metric]
            lo, hi = ci(boot[a][metric] - boot[b][metric])
            rows_out.append({"contrast": f"{a} minus {b}", "metric": metric,
                "value_group_a": points[a][metric], "value_group_b": points[b][metric],
                "difference": delta, "ci_low": lo, "ci_high": hi,
                "n_subjects_a": points[a]["n_subjects"],
                "n_subjects_b": points[b]["n_subjects"],
                "n_nights_a": points[a]["n_nights"], "n_nights_b": points[b]["n_nights"],
                "prevalence_a": points[a]["prevalence"],
                "prevalence_b": points[b]["prevalence"]})
    return pd.DataFrame(rows_out)


def bootstrap_miss_rate(d: pd.DataFrame, n_boot: int, seed: int):
    per = d.groupby(PID).agg(missed=("missed", "sum"), positive=("missed", "size"))
    point = float(per.missed.sum() / per.positive.sum())
    rng = np.random.default_rng(seed)
    arr_m = per.missed.to_numpy()
    arr_n = per.positive.to_numpy()
    draw = rng.integers(0, len(per), size=(n_boot, len(per)))
    rates = arr_m[draw].sum(axis=1) / arr_n[draw].sum(axis=1)
    return point, rates, len(per), int(per.positive.sum()), int(per.missed.sum())


def summarize_miss_rates(pos: pd.DataFrame, n_boot: int, seed: int):
    sets = [("All", pos)] + [(str(g), d) for g, d in pos.groupby("age_group")] + [
        (c, pos[pos.cohort == c]) for c in ("children", "teens", "adults")]
    rows, boots = [], {}
    for i, (name, d) in enumerate(sets):
        if d.empty:
            continue
        point, b, ns, nn, nm = bootstrap_miss_rate(d, n_boot, seed + 1000 + i)
        lo, hi = ci(b)
        rows.append({"group": name, "n_subjects": ns, "n_positive_nights": nn,
                     "n_missed_nights": nm, "miss_rate": point, "ci_low": lo, "ci_high": hi})
        boots[name] = b
    if "children" in boots and "adults" in boots:
        ch = next(r for r in rows if r["group"] == "children")
        ad = next(r for r in rows if r["group"] == "adults")
        delta = ch["miss_rate"] - ad["miss_rate"]
        lo, hi = ci(boots["children"] - boots["adults"])
        rows.append({"group": "children minus adults", "n_subjects": np.nan,
                     "n_positive_nights": np.nan, "n_missed_nights": np.nan,
                     "miss_rate": delta, "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows), boots


def subject_factor_tables(pos: pd.DataFrame):
    """Return long subject/status summaries and one positive-night summary per subject."""
    long_rows, all_rows = [], []
    for pid, d in pos.groupby(PID, sort=False):
        base = {PID: pid, "age_group": d.age_group.iloc[0], "cohort": d.cohort.iloc[0],
                "source_dataset": d.source_dataset.iloc[0],
                "n_positive_nights": len(d), "miss_rate": float(d.missed.mean())}
        all_row = dict(base)
        for f in FACTORS:
            all_row[f] = d[f].median()
        all_rows.append(all_row)
        for status, ds in d.groupby("status"):
            row = {PID: pid, "age_group": base["age_group"], "cohort": base["cohort"],
                   "status": status, "n_positive_nights": len(ds)}
            for f in FACTORS:
                row[f] = ds[f].median()
            long_rows.append(row)
    return pd.DataFrame(long_rows), pd.DataFrame(all_rows)


def paired_missed_caught(subject_status: pd.DataFrame, n_boot: int, seed: int):
    groups = [("All", subject_status)] + list(subject_status.groupby("age_group"))
    rows = []
    for gi, (name, d) in enumerate(groups):
        for fi, (f, label) in enumerate(FACTORS.items()):
            wide = d.pivot(index=PID, columns="status", values=f)
            if "missed" not in wide or "caught" not in wide:
                paired = pd.DataFrame(columns=["missed", "caught"])
            else:
                paired = wide[["missed", "caught"]].dropna()
            delta = (paired.missed - paired.caught).to_numpy(float) if len(paired) else np.array([])
            if len(delta):
                point = float(np.median(delta))
                rng = np.random.default_rng(seed + gi * 100 + fi)
                b = np.median(delta[rng.integers(0, len(delta), size=(n_boot, len(delta)))], axis=1)
                lo, hi = ci(b)
                p = float(wilcoxon(delta).pvalue) if len(delta) >= 5 and np.any(delta != 0) else np.nan
                med_miss, med_caught = float(paired.missed.median()), float(paired.caught.median())
            else:
                point = lo = hi = med_miss = med_caught = p = np.nan
            rows.append({"group": name, "factor": f, "factor_label": label,
                         "n_subjects_paired": len(delta), "median_subject_value_missed": med_miss,
                         "median_subject_value_caught": med_caught,
                         "median_within_subject_difference_missed_minus_caught": point,
                         "ci_low": lo, "ci_high": hi, "p": p})
    out = pd.DataFrame(rows)
    out["p_holm"] = np.nan
    for _, ix in out.groupby("group").groups.items():
        out.loc[ix, "p_holm"] = holm(out.loc[ix, "p"])
    return out


def child_adult_patterns(subject_all: pd.DataFrame, n_boot: int, seed: int):
    ca = subject_all[subject_all.cohort.isin(["children", "adults"])]
    rows = []
    for fi, (f, label) in enumerate(FACTORS.items()):
        x = ca.loc[ca.cohort == "children", f].dropna().to_numpy(float)
        y = ca.loc[ca.cohort == "adults", f].dropna().to_numpy(float)
        if len(x) >= 5 and len(y) >= 5:
            u, p = mannwhitneyu(x, y, alternative="two-sided")
            effect = 2 * u / (len(x) * len(y)) - 1
            point = float(np.median(x) - np.median(y))
            rng = np.random.default_rng(seed + 3000 + fi)
            bx = np.median(x[rng.integers(0, len(x), size=(n_boot, len(x)))], axis=1)
            by = np.median(y[rng.integers(0, len(y), size=(n_boot, len(y)))], axis=1)
            lo, hi = ci(bx - by)
        else:
            u = p = effect = point = lo = hi = np.nan
        rows.append({"factor": f, "factor_label": label, "n_children_subjects": len(x),
                     "n_adult_subjects": len(y), "median_children": np.median(x) if len(x) else np.nan,
                     "median_adults": np.median(y) if len(y) else np.nan,
                     "median_difference_children_minus_adults": point,
                     "ci_low": lo, "ci_high": hi, "rank_biserial_r": effect, "p": p})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out.p)
    return out


def subject_feature_missrate(subject_all: pd.DataFrame):
    rows = []
    groups = [("All", subject_all)] + list(subject_all.groupby("age_group"))
    for name, d in groups:
        for f, label in FACTORS.items():
            x = d[f].to_numpy(float)
            y = d.miss_rate.to_numpy(float)
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() >= 5 and np.std(x[ok]) > 0 and np.std(y[ok]) > 0:
                rho, p = spearmanr(x[ok], y[ok])
            else:
                rho = p = np.nan
            rows.append({"group": name, "factor": f, "factor_label": label,
                         "n_subjects": int(ok.sum()), "spearman_rho": rho, "p": p})
    out = pd.DataFrame(rows)
    out["p_holm"] = np.nan
    for _, ix in out.groupby("group").groups.items():
        out.loc[ix, "p_holm"] = holm(out.loc[ix, "p"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="n00-06_lt70_r4")
    ap.add_argument("--suffix-extra", default="", help="e.g. _no-SOURCE")
    ap.add_argument("--model", choices=["auto", "logreg", "hgb", "cnn"], default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--results", type=Path, default=Path("results"))
    args = ap.parse_args()
    suffix = f"_{args.tag}{args.suffix_extra}"
    out = args.results / "error_analysis" / suffix.lstrip("_")
    out.mkdir(parents=True, exist_ok=True)

    model, scores = (args.model, {}) if args.model != "auto" else choose_model(args.results, suffix)
    pred_file = (args.results / f"cnn_test_predictions{suffix}.csv" if model == "cnn"
                 else args.results / f"baseline_test_predictions{suffix}.csv")
    P = pd.read_csv(pred_file, dtype={PID: str})
    prediction_model = "cnn_5seed_avg" if model == "cnn" else model
    P = P[(P.model == prediction_model) & (P.scope == "global")].copy()
    if P.empty:
        raise SystemExit(f"No global test predictions for selected model {model}: {pred_file}")
    print(f"Best model by validation AUROC: {model} (" + ", ".join(f"{k}={v:.4f}" for k, v in scores.items()) + ")")

    test = pd.read_csv(args.results / "features" / f"features_{args.tag}_test.csv", dtype={PID: str})
    test = test.merge(P[["night_id", "prob", "pred", "threshold"]],
                      on="night_id", how="inner", validate="one_to_one")
    if not (test.label.to_numpy() == P.set_index("night_id").loc[test.night_id, "label"].to_numpy()).all():
        raise ValueError("Prediction labels do not match the frozen test feature labels.")
    test["cohort"] = test.age_group.map(cohort)
    pos = test[test.label == 1].copy()
    pos["missed"] = (pos.pred == 0).astype(int)
    pos["status"] = np.where(pos.missed == 1, "missed", "caught")
    pos["evening_missing_pct"] = 100 * (1 - pos.evening_coverage)

    # Primary age contrast and subject-clustered miss rates.
    gap = primary_age_gap(test[[PID, "age_group", "label", "prob"]], args.n_boot, args.seed)
    pd.DataFrame([gap]).round(5).to_csv(out / "primary_auroc_difference.csv", index=False)
    pairwise = pairwise_age_differences(test[[PID, "age_group", "label", "prob"]],
                                        args.n_boot, args.seed)
    pairwise.round(5).to_csv(out / "pairwise_age_differences.csv", index=False)
    MR, _ = summarize_miss_rates(pos, args.n_boot, args.seed)
    MR.round(5).to_csv(out / "miss_rates.csv", index=False)

    # One median per subject/status for within-person missed-versus-caught checks.
    subject_status, subject_all = subject_factor_tables(pos)
    subject_status.to_csv(out / "subject_factors_by_status.csv", index=False)
    subject_all.to_csv(out / "positive_subject_summary.csv", index=False)
    paired = paired_missed_caught(subject_status, args.n_boot, args.seed)
    paired.round(5).to_csv(out / "missed_vs_caught.csv", index=False)
    age_patterns = child_adult_patterns(subject_all, args.n_boot, args.seed)
    age_patterns.round(5).to_csv(out / "child_vs_adult.csv", index=False)
    corr = subject_feature_missrate(subject_all)
    corr.round(5).to_csv(out / "subject_feature_missrate.csv", index=False)

    # Post-midnight outcomes are descriptors only, never predictors of errors.
    event_rows = []
    for name, d in [("All", pos)] + list(pos.groupby("age_group")):
        for field in ("onset_min_after_start", "night_nadir"):
            v = d[field].dropna().to_numpy(float)
            event_rows.append({"group": name, "descriptor": field, "n_nights": len(v),
                               "median": np.median(v) if len(v) else np.nan,
                               "q1": np.percentile(v, 25) if len(v) else np.nan,
                               "q3": np.percentile(v, 75) if len(v) else np.nan})
    pd.DataFrame(event_rows).to_csv(out / "event_descriptors.csv", index=False)

    mr = MR.set_index("group")
    if gap["ci_high"] < 0:
        gap_text = "children had a lower test AUROC than adults (the 95% CI was below zero)"
    elif gap["ci_low"] > 0:
        gap_text = "children had a higher test AUROC than adults (the 95% CI was above zero)"
    else:
        gap_text = "the test AUROC difference was not clearly different from zero"
    miss_delta = mr.loc["children minus adults", "miss_rate"]
    miss_lo = mr.loc["children minus adults", "ci_low"]
    miss_hi = mr.loc["children minus adults", "ci_high"]
    paired_all = paired[paired.group == "All"].dropna(subset=["p_holm"]).sort_values("p_holm")
    top = paired_all.iloc[0] if len(paired_all) else None
    lines = [
        f"Selected global model: {model}, chosen by validation AUROC only ({scores}).",
        "The pre-specified primary contrast is children (0-13) minus adults (>=18), with independent subject-level bootstrap resampling.",
        f"Primary AUROC: children {gap['auroc_children']:.3f} ({gap['n_subjects_children']} subjects, {gap['n_positive_children']} positive nights); adults {gap['auroc_adults']:.3f} ({gap['n_subjects_adults']} subjects, {gap['n_positive_adults']} positive nights).",
        f"Children minus adults = {gap['children_minus_adults']:+.3f} (95% CI {gap['ci_low']:+.3f} to {gap['ci_high']:+.3f}); {gap_text}.",
        f"Miss rate difference (children minus adults) = {miss_delta:+.3f} (95% CI {miss_lo:+.3f} to {miss_hi:+.3f}).",
        "Pattern tests use one subject summary per person and only predictors available before midnight. They are exploratory associations, not causal explanations.",
        "Post-midnight onset and nadir are listed only as event descriptors; they are not used to explain bedtime prediction errors.",
    ]
    if top is not None:
        lines.append(f"Smallest Holm-adjusted p-value in the exploratory within-subject missed-versus-caught checks: {top.factor}, with median difference {top.median_within_subject_difference_missed_minus_caught:+.3f} (Holm p={top.p_holm:.3g}, {int(top.n_subjects_paired)} paired subjects). This does not establish an age-gap mechanism.")
    summary = "\n".join(lines) + "\n"
    (out / "summary.txt").write_text(summary)
    key_factor = str(top.factor) if top is not None else "cv"
    (out / "chosen.json").write_text(json.dumps({"model": model, "val_scores": scores,
        "key_factor": key_factor, "primary_age_gap": gap}, indent=2))
    print("\n" + summary)
    print(f"Saved subject-level tables to {out}/")


if __name__ == "__main__":
    main()
