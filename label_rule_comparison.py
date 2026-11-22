"""How many nights change label between two labelling rules, by age group and source.

Usage:
  python src/label_rule_comparison.py --a n00-06_lt70_r4 --b n00-06_lt70_r4_i4
  python src/label_rule_comparison.py --a n00-06_lt70_r4 --b n00-06_le70_r3
Output: results/label_change_<a>_vs_<b>.csv (and printed). Only nights present
under BOTH rules are compared; the counts of nights present in only one are printed.
"""
import argparse
from pathlib import Path

import pandas as pd

KEY = ["PtID", "night_date"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--results", type=Path, default=Path("results"))
    args = ap.parse_args()
    cols = KEY + ["age_group", "source_dataset", "label", "label_ambiguous_gap"]
    A = pd.read_csv(args.results / f"nights_{args.a}.csv", dtype={"PtID": str})
    B = pd.read_csv(args.results / f"nights_{args.b}.csv", dtype={"PtID": str})
    A = A[[c for c in cols if c in A]]
    B = B[[c for c in KEY + ["label", "label_ambiguous_gap"] if c in B]]
    m = A.merge(B, on=KEY, suffixes=("_a", "_b"), how="inner")
    print(f"nights only in {args.a}: {len(A) - len(m)}; only in {args.b}: {len(B) - len(m)}")
    m["change"] = (m["label_a"].astype(str) + "->" + m["label_b"].astype(str))
    rows = []
    for by in ("age_group", "source_dataset"):
        t = m.groupby(by).agg(n_nights=("label_a", "size"),
                              n_pos_a=("label_a", "sum"), n_pos_b=("label_b", "sum"),
                              neg_to_pos=("change", lambda c: int((c == "0->1").sum())),
                              pos_to_neg=("change", lambda c: int((c == "1->0").sum())))
        t.insert(0, "grouped_by", by)
        rows.append(t.reset_index().rename(columns={by: "group"}))
    out = pd.concat(rows, ignore_index=True)
    out["pct_changed"] = (100 * (out["neg_to_pos"] + out["pos_to_neg"]) / out["n_nights"]).round(2)
    out.to_csv(args.results / f"label_change_{args.a}_vs_{args.b}.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
