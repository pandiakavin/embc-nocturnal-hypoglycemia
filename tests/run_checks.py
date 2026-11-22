"""Run the whole pipeline on FAKE data and assert the known answers.
Usage (from the project folder):  python tests/run_checks.py
"""
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))
from nights import count_episodes_recovery
from reading_carry_forward_audit import audit as audit_repeated_readings

t = root / "tests"
raw = t / "fake_raw"
for output_dir in (t / "out_results", t / "out_processed"):
    shutil.rmtree(output_dir, ignore_errors=True)
sh = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()

# Recovery episodes require observed non-low samples. Two observed highs are
# insufficient to separate episodes; three are sufficient; a missing sample
# cannot count as recovery.
episode_low = np.zeros((3, 12), dtype=bool)
episode_observed = np.ones((3, 12), dtype=bool)
episode_low[0, 0:4] = episode_low[0, 6:10] = True
episode_low[1, 0:4] = episode_low[1, 7:11] = True
episode_low[2, 0:4] = episode_low[2, 5:9] = True
episode_observed[2, 4] = False
assert count_episodes_recovery(episode_low, episode_observed, min_run=4, recovery=3).tolist() == [1, 2, 1]

subprocess.run([sys.executable, t / "make_fake_data.py"], check=True)
# Repeated-reading audit counts observed exact 5-minute pairs, including a
# duplicate that crosses a CSV chunk boundary; missing and 10-minute pairs do not count.
audit_input = t / "carry_audit.csv"
pd.DataFrame([
    ("2020-01-01T00:00:00", "A", 100.0, "X"),
    ("2020-01-01T00:05:00", "A", 100.0, "X"),
    ("2020-01-01T00:10:00", "A", 100.0, "X"),
    ("2020-01-01T00:15:00", "A", np.nan, "X"),
    ("2020-01-01T00:20:00", "A", 100.0, "X"),
    ("2020-01-01T00:25:00", "A", 100.0, "X"),
    ("2020-01-01T00:35:00", "A", 100.0, "X"),
], columns=["ts", "PtID", "GlucoseCGM", "Database"]).to_csv(audit_input, index=False)
audit_result = audit_repeated_readings(audit_input, chunksize=2).iloc[0]
assert audit_result.observed_adjacent_5min_pairs == 3
assert audit_result.exactly_repeated_pairs == 3
before = {p.name: sh(p) for p in raw.glob("*.csv")}
common = ["--main", raw / "maindatabase.csv", "--demographics", raw / "demographics.csv"]
subprocess.run([sys.executable, root / "src/explore_data.py", *common,
                "--out", t / "out_results"], check=True, stdout=subprocess.DEVNULL)
for extra in (["--op", "le", "--min-run", "3"], ["--op", "lt", "--min-run", "3"], [],
              ["--interp-max-missing", "4"]):
    subprocess.run([sys.executable, root / "src/nights.py", *common, *extra,
                    "--results", t / "out_results", "--processed", t / "out_processed"],
                   check=True, stdout=subprocess.DEVNULL)

s = pd.read_csv(t / "out_results/data_summary.csv").set_index("age_group")
assert s.loc["All", "n_subjects"] == 44
assert s.loc["All", "n_age_range_only"] == 5          # k = 0, 10, 20, 30, 40
assert s.loc["adult 18+", "n_subjects"] >= 4           # range 60-100 fits adult 18+
sub = pd.read_csv(t / "out_results/subject_summary.csv").set_index("PtID")
demo_rows = pd.read_csv(raw / "demographics.csv")
k3 = demo_rows["PtID"].drop_duplicates().iloc[3]
assert sub.loc[k3, "age_group"] == "child 0-13", "age at first reading (13) must decide the group"

for fname, expect in [("nights_n00-06_le70_r3.csv", {"S_pos3": 1, "S_neg2": 0, "S_gap": 0, "S_eq70": 1}),
                      ("nights_n00-06_lt70_r3.csv", {"S_pos3": 1, "S_eq70": 0}),
                      # primary rule: <70 for 4 readings -> 3 readings is NOT enough
                      ("nights_n00-06_lt70_r4.csv", {"S_pos3": 0, "S_pos4": 1, "S_eq70": 0,
                          "S_recent_low": 1, "S_cross_midnight": 1, "S_end_onset": 1,
                          "S_gap_all": 0})]:
    d = pd.read_csv(t / "out_results" / fname)
    d = d[d.night_date == "2020-01-02"].set_index("PtID")
    for pid, lab in expect.items():
        assert d.loc[pid, "label"] == lab, (fname, pid)
    assert "S_lowcov" not in d.index, "low-coverage night should be dropped"
d = pd.read_csv(t / "out_results/nights_n00-06_le70_r3.csv").query("night_date == '2020-01-02'").set_index("PtID")
assert d.loc["S_gap", "label_ambiguous_gap"] == 1 and d.loc["S_neg2", "label_ambiguous_gap"] == 0

subprocess.run([sys.executable, root / "src/features.py", "--tag", "n00-06_le70_r3",
                "--results", t / "out_results", "--processed", t / "out_processed"],
               check=True, stdout=subprocess.DEVNULL)
splits = pd.read_csv(t / "out_results/splits_seed42.csv")
assert splits["PtID"].is_unique
parts = {sp: pd.read_csv(t / f"out_results/features/features_n00-06_le70_r3_{sp}.csv") for sp in ("train", "val", "test")}
ids = [set(p["PtID"]) for p in parts.values()]
assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]), "subject leak across splits"
f = pd.concat(parts.values()).query("night_date == '2020-01-02'").set_index("PtID")
r = f.loc["S_slope"]
assert r.last_glucose == 129 and r["min"] == 129 and abs(r["mean"] - 164.5) < 1e-6
assert abs(r.roc_30 + 0.2) < 1e-9 and abs(r.roc_60 + 0.2) < 1e-9
assert abs(r.pct_above_180 - 100 * 20 / 72) < 1e-6 and r.pct_below_70 == 0
r = f.loc["S_lowev"]
assert abs(r.pct_below_70 - 100 * 12 / 72) < 1e-6 and r.prior24h_low_episodes == 1
r = f.loc["S_pos3"]
assert r.sd == 0 and r.cv == 0 and r.roc_30 == 0
# a re-run must reuse the same split file (same assignments)
subprocess.run([sys.executable, root / "src/features.py", "--tag", "n00-06_le70_r3",
                "--results", t / "out_results", "--processed", t / "out_processed"],
               check=True, stdout=subprocess.DEVNULL)
assert pd.read_csv(t / "out_results/splits_seed42.csv").equals(splits)

# primary: S_gap (65, missing, 65, 65 + one more low) is ambiguous; filling the
# short gap (sensitivity) makes it positive; features.py drops ambiguous by default
n = pd.read_csv(t / "out_results/nights_n00-06_lt70_r4.csv").query("night_date == '2020-01-02'").set_index("PtID")
ni = pd.read_csv(t / "out_results/nights_n00-06_lt70_r4_i4.csv").query("night_date == '2020-01-02'").set_index("PtID")
assert n.loc["S_gap4", "label"] == 0 and n.loc["S_gap4", "label_ambiguous_gap"] == 1
assert n.loc["S_gap_all", "label"] == 0 and n.loc["S_gap_all", "label_ambiguous_gap"] == 1
assert ni.loc["S_gap4", "label"] == 1
assert ni.loc["S_gap_all", "label"] == 0 and ni.loc["S_gap_all", "label_ambiguous_gap"] == 0
assert ni.loc["S_neg2", "label"] == 0 and ni.loc["S_pos4", "label"] == 1
subprocess.run([sys.executable, root / "src/label_rule_comparison.py", "--a", "n00-06_lt70_r4",
                "--b", "n00-06_lt70_r4_i4", "--results", t / "out_results"], check=True,
               stdout=subprocess.DEVNULL)
lc = pd.read_csv(t / "out_results/label_change_n00-06_lt70_r4_vs_n00-06_lt70_r4_i4.csv")
assert lc.query("grouped_by == 'age_group'")["neg_to_pos"].sum() >= 1
# primary tag: ongoing-at-midnight low is labelled, flagged, and dropped by features.py
n = pd.read_csv(t / "out_results/nights_n00-06_lt70_r4.csv").query("night_date == '2020-01-02'").set_index("PtID")
assert n.loc["S_ongoing", "label"] == 1 and n.loc["S_ongoing", "low_in_progress_at_start"] == 1
assert n.loc["S_cross_midnight", "label"] == 1 and n.loc["S_cross_midnight", "low_in_progress_at_start"] == 1
assert n.loc["S_cross_midnight", "onset_min_after_start"] == -5
assert n.loc["S_recent_low", "label"] == 1 and n.loc["S_recent_low", "low_in_progress_at_start"] == 0
assert n.loc["S_end_onset", "label"] == 1 and n.loc["S_end_onset", "onset_min_after_start"] == 355
assert n.loc["S_pos4", "low_in_progress_at_start"] == 0
subprocess.run([sys.executable, root / "src/features.py", "--results", t / "out_results",
                "--processed", t / "out_processed"], check=True, stdout=subprocess.DEVNULL)
fp = pd.concat(pd.read_csv(t / f"out_results/features/features_n00-06_lt70_r4_{sp}.csv")
               for sp in ("train", "val", "test")).query("night_date == '2020-01-02'")
assert "S_ongoing" not in set(fp["PtID"]) and "S_pos4" in set(fp["PtID"])
assert "S_cross_midnight" not in set(fp["PtID"])
assert "S_recent_low" in set(fp["PtID"]) and "S_end_onset" in set(fp["PtID"])
assert "S_gap4" not in set(fp["PtID"]), "ambiguous-gap negative must be dropped in the primary run"
assert "S_gap_all" not in set(fp["PtID"]), "all-missing event-length gap must be unknown in the primary run"

assert before == {p.name: sh(p) for p in raw.glob("*.csv")}, "raw files changed!"
print("ALL CHECKS PASSED")
