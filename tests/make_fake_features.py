"""FAKE feature tables with a planted signal, for testing baselines.py only.

Children (0-13) get a WEAKER signal (more random) than other groups, so a
correct pipeline should show lower AUROC for 0-13. Seed 0. Not real data.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from features import FEATURE_COLS  # noqa: E402

out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "fake_feat_results"
(out / "features").mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)
groups = ["0-13", "14-24", "25-44", "45-100"]
rows, nid = [], 0
for k in range(400):
    g = groups[k % 4]
    split = ["train"] * 14 + ["val"] * 3 + ["test"] * 3
    sp = split[(k // 4) % 20]
    subj_offset = rng.normal(0, 15)
    for n in range(25):
        last = rng.normal(130 + subj_offset, 35)
        noise = 2.5 if g == "0-13" else 0.8
        logit = -0.04 * (last - 100) - 1.0 + rng.normal(0, noise)
        y = int(rng.random() < 1 / (1 + np.exp(-logit)))
        f = {c: rng.normal(0, 1) for c in FEATURE_COLS}
        f["last_glucose"] = last
        if rng.random() < 0.05:
            f["roc_30"] = np.nan
        rows.append({"night_id": nid, "PtID": f"P{k:03d}", "age_group": g, "split": sp,
                     "label": y, "low_in_progress_at_start": 0, **f})
        nid += 1
df = pd.DataFrame(rows)
for sp in ("train", "val", "test"):
    df[df.split == sp].to_csv(out / "features" / f"features_n00-06_lt70_r4_{sp}.csv", index=False)
print(f"positives: {df.label.mean():.1%}")
