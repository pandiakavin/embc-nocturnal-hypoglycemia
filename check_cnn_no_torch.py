"""Tests everything in cnn.py EXCEPT the PyTorch training loop, by swapping in
a simple stand-in model. Works without torch installed.
Usage: python tests/check_cnn_no_torch.py
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))
import cnn  # noqa: E402

# 1) interpolation + scaling
ev = np.array([[100, np.nan, 120, np.nan], [np.nan, 50, 50, np.nan]], dtype="float32")
f = cnn.interpolate_rows(ev)
assert np.allclose(f, [[100, 110, 120, 120], [50, 50, 50, 50]])
X = cnn.build_inputs({"train": ev, "val": ev, "test": ev})
assert X["train"].shape == (2, 2, 4) and X["train"][0, 1].tolist() == [0, 1, 0, 1]

# 2) full pipeline on fake data, with a stand-in "model"
tmp = Path(tempfile.mkdtemp())
subprocess.run([sys.executable, root / "tests/make_fake_features.py", tmp], check=True)
# build a windows file aligned with the fake features: evening ends at last_glucose
frames = [pd.read_csv(tmp / "features" / f"features_n00-06_lt70_r4_{s}.csv")
          for s in ("train", "val", "test")]
allf = pd.concat(frames).sort_values("night_id")
rng = np.random.default_rng(1)
evening = (allf["last_glucose"].to_numpy()[:, None]
           + rng.normal(0, 5, (len(allf), 72))).astype("float32")
evening[rng.random(evening.shape) < 0.05] = np.nan
(tmp / "processed").mkdir()
np.savez(tmp / "processed" / "windows_n00-06_lt70_r4.npz",
         night_id=allf["night_id"].to_numpy(), evening=evening)
subprocess.run([sys.executable, root / "src/baselines.py", "--results", tmp, "--n-boot", "100"],
               check=True, stdout=subprocess.DEVNULL)


def stand_in(seed, X, y, args):
    """Not a CNN: score = minus mean of last 3 scaled glucose values + seed noise."""
    r = np.random.default_rng(seed)
    score = lambda x: -x[:, 0, -3:].mean(1) + r.normal(0, 0.05, len(x))
    sig = lambda z: 1 / (1 + np.exp(-z))
    return sig(score(X["val"])), sig(score(X["test"])), [{"seed": seed, "epoch": 0, "val_auroc": 0.8}]


cnn.main(["--results", str(tmp), "--processed", str(tmp / "processed"), "--n-boot", "100"],
         fit=stand_in)
res = pd.read_csv(tmp / "cnn_results.csv").set_index("eval_group")
comp = pd.read_csv(tmp / "cnn_vs_baselines.csv")
seeds = pd.read_csv(tmp / "cnn_results_seeds.csv")
assert set(res.index) == {"All", "0-13", "14-24", "25-44", "45-100"}
assert res.loc["All", "auroc"] > 0.7, "stand-in should recover the planted signal"
assert res.loc["0-13", "auroc"] < res.loc["45-100", "auroc"], "children were planted harder"
assert (res["auroc_ci_low"] <= res["auroc"]).all() and (res["auroc"] <= res["auroc_ci_high"]).all()
assert len(seeds[seeds.seed != "mean/sd"]) == 5 * 5
assert set(comp["verdict"]) <= {"CNN better", "CNN worse", "no clear difference"}
assert (tmp / "cnn_test_predictions_n00-06_lt70_r4.csv").exists()
assert (tmp / "cnn_val_scores_n00-06_lt70_r4.csv").exists()
assert (tmp / "cnn_validation_predictions_n00-06_lt70_r4.csv").exists()
v = pd.read_csv(tmp / "cnn_validation_predictions_n00-06_lt70_r4.csv")
scores = pd.read_csv(tmp / "cnn_val_scores_n00-06_lt70_r4.csv")
ensemble = scores[scores.score_type == "five_seed_average_used_for_selection"]
assert len(ensemble) == 1
assert np.isclose(ensemble.iloc[0].best_val_auroc, roc_auc_score(v.label, v.prob))
print("CNN PIPELINE CHECKS PASSED (training loop itself needs torch - run on your computer)")
