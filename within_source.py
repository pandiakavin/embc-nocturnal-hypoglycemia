"""POST HOC sensitivity (added 2026-11-16 after the source-composition audit):
compare age groups only WITHIN the same source dataset, so differences between
source studies (device, protocol, population) cannot masquerade as age effects.

For every source dataset whose TEST set contains >= 2 age groups, and for the
chosen model's global predictions, report AUROC per age group with subject-level
bootstrap 95% CIs and the difference to that source's adult group.
Groups below the analysis-plan minimum (>= 10 test subjects AND >= 20 positive
nights) are reported but flagged 'below_min_data' - describe them, don't test them.

Usage
  python src/within_source.py --model hgb          # or logreg / cnn
Output
  results/within_source_<tag>_<model>.csv
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import diadata_io as io
from baselines import SubjectBootstrap, ci

PID = "PtID"


def lower(g):
    m = re.search(r"\d+", str(g))
    return float(m.group()) if m else np.nan


def boot_auc(d, n_boot, seed):
    y, p = d["label"].to_numpy(), d["prob"].to_numpy()
    bs = SubjectBootstrap(d[PID].to_numpy(), n_boot, seed)
    b = [roc_auc_score(y[i], p[i]) if len(np.unique(y[i])) == 2 else np.nan
         for i in bs.indices()]
    pt = roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan
    return pt, np.array(b), len(bs.uniq)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default=io.PRIMARY_TAG)
    ap.add_argument("--model", choices=["logreg", "hgb", "cnn"], required=True)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--results", type=Path, default=Path("results"))
    args = ap.parse_args()
    R = args.results
    if args.model == "cnn":
        P = pd.read_csv(R / f"cnn_test_predictions_{args.tag}.csv", dtype={PID: str})
    else:
        P = pd.read_csv(R / f"baseline_test_predictions_{args.tag}.csv", dtype={PID: str})
        P = P[(P.model == args.model) & (P.scope == "global")]
    test = pd.read_csv(R / "features" / f"features_{args.tag}_test.csv", dtype={PID: str})
    d = test[["night_id", PID, "age_group", "source_dataset", "label"]].merge(
        P[["night_id", "prob"]], on="night_id", how="inner")
    assert len(d) == len(test), "predictions do not cover every test night"

    rows = []
    for source_index, (src, s) in enumerate(d.groupby("source_dataset", sort=True)):
        groups = sorted(s["age_group"].unique(), key=lower)
        if len(groups) < 2:
            continue
        adults = [g for g in groups if lower(g) >= 18]
        ref = adults[0] if adults else groups[-1]
        # Each source/group gets an independent, deterministic subject-bootstrap
        # stream. The adult reference draw is then shared only within its paired
        # contrasts against other age groups from this same source.
        res = {
            g: boot_auc(s[s.age_group == g], args.n_boot,
                        args.seed + 1000 * source_index + group_index + 1)
            for group_index, g in enumerate(groups)
        }
        for g in groups:
            sg = s[s.age_group == g]
            pt, b, n_subj = res[g]
            lo, hi = ci(b)
            row = {"source_dataset": src, "age_group": g, "n_subjects": n_subj,
                   "n_nights": len(sg), "n_positive": int(sg.label.sum()),
                   "auroc": pt, "ci_low": lo, "ci_high": hi,
                   "below_min_data": bool(n_subj < 10 or sg.label.sum() < 20),
                   "reference_group": ref}
            if g != ref:
                dpt = pt - res[ref][0]
                dlo, dhi = ci(b - res[ref][1])
                row.update({"diff_vs_reference": dpt, "diff_ci_low": dlo, "diff_ci_high": dhi})
            rows.append(row)
    out = pd.DataFrame(rows)
    path = R / f"within_source_{args.tag}_{args.model}.csv"
    out.round(4).to_csv(path, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(out.round(3).to_string(index=False))
    print(f"\nPOST HOC. Saved {path}")


if __name__ == "__main__":
    main()
