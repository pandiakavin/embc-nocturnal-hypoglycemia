"""Make small FAKE DiaData-shaped CSVs with nights whose correct labels are known.

This is only for testing the code. Nothing here is real data.
Subjects:
  S_pos3     night 2020-01-02: exactly 3 readings at 65 from 02:00 -> POSITIVE
  S_neg2     night 2020-01-02: only 2 readings at 65 -> NEGATIVE
  S_gap      night 2020-01-02: 65, missing, 65, 65 ... gap breaks run -> only
             2 consecutive -> NEGATIVE
  S_lowcov   night 2020-01-02 has only 75% of readings -> DROPPED
  S_eq70     night 2020-01-02: 3 readings of exactly 70 -> POSITIVE with <=70,
             NEGATIVE with <70
  S_slope    evening before 2020-01-02 falls 200 -> 129 by 1 per reading
  S_lowev    evening before 2020-01-02 starts with 1 h at 60 mg/dL
  R_00..R_29 random subjects (seed 0) across 3 source datasets
"""
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
out = Path(__file__).parent / "fake_raw"
out.mkdir(exist_ok=True)

grid = pd.date_range("2020-01-01 00:00", "2020-01-04 23:55", freq="5min")


def base_series():
    return np.full(len(grid), 140.0)


def night_idx(time_str):
    return int(np.flatnonzero(grid == pd.Timestamp(f"2020-01-02 {time_str}"))[0])


rows = []
specs = {}
g = base_series(); i = night_idx("02:00"); g[i:i + 3] = 65; specs["S_pos3"] = (g, "A")
g = base_series(); g[i:i + 2] = 65; specs["S_neg2"] = (g, "A")
g = base_series(); g[i] = 65; g[i + 1] = np.nan; g[i + 2:i + 4] = 65; specs["S_gap4"] = (g, "C")   # 1 gap inside 4 lows
g = base_series(); g[i:i + 4] = 65; specs["S_pos4"] = (g, "C")   # 4 readings: positive under the primary rule
j0 = night_idx("00:00")
g = base_series(); g[j0 - 3:j0 + 6] = 60; specs["S_ongoing"] = (g, "A")   # low from 23:45 into the night
g = base_series(); g[j0 - 1:j0 + 3] = 60; specs["S_cross_midnight"] = (g, "A")  # qualifying run begins at 23:55
g = base_series(); g[night_idx("23:45")] = 65; g[j0:j0 + 4] = 65
specs["S_recent_low"] = (g, "A")  # earlier isolated low; qualifying event starts at 00:00
g = base_series(); j = night_idx("05:55"); g[j:j + 4] = 65
specs["S_end_onset"] = (g, "A")  # event starts before 06:00, confirmed at 06:10
g = base_series(); j = night_idx("02:00"); g[j:j + 4] = np.nan
specs["S_gap_all"] = (g, "A")  # a full unobserved event-length interval
g = base_series(); g[i] = 65; g[i + 1] = np.nan; g[i + 2:i + 4] = 65; specs["S_gap"] = (g, "B")
g = base_series(); j = night_idx("00:00"); g[j:j + 18] = np.nan; specs["S_lowcov"] = (g, "B")  # 18/72 = 25% missing
g = base_series(); g[i:i + 3] = 70; specs["S_eq70"] = (g, "C")
# evening before night 2020-01-02 = 2020-01-01 18:00-23:55 (72 readings)
ev0 = int(np.flatnonzero(grid == pd.Timestamp("2020-01-01 18:00"))[0])
g = base_series(); g[ev0:ev0 + 72] = 200 - np.arange(72); specs["S_slope"] = (g, "A")   # 200 -> 129, -0.2 mg/dL/min
g = base_series(); g[ev0:ev0 + 12] = 60; specs["S_lowev"] = (g, "B")   # 1 h at 60 -> 16.67% below 70, 1 episode

for pid, (g, db) in specs.items():
    for t, v in zip(grid, g):
        if not np.isnan(v):
            rows.append((t, pid, v, db))

for k in range(30):
    pid = f"R_{k:02d}"
    db = "ABC"[k % 3]
    g = 150 + 40 * np.sin(np.arange(len(grid)) / 40 + k) + rng.normal(0, 15, len(grid))
    g[rng.random(len(grid)) < 0.1] = np.nan          # ~10% missing at random
    for t, v in zip(grid, g):
        if not np.isnan(v):
            rows.append((t, pid, round(float(v), 1), db))

main = pd.DataFrame(rows, columns=["ts", "PtID", "GlucoseCGM", "Database"])
main["ts"] = main["ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
main.to_csv(out / "maindatabase.csv", index=False)

pids = list(specs) + [f"R_{k:02d}" for k in range(30)]
# DiaData v3 Subdatabase I layout: ts,PtID,GlucoseCGM,Age,Sex,Database, Age = age AT
# EACH READING. Two rows per subject: at start and one year later.
rep_ages = [1, 4, 8, 12, 15, 20, 30, 40, 60]
rows_d = []
for k, pid in enumerate(pids):
    db = "ABC"[k % 3]
    if k % 10 == 0:                      # range-only subject (like SHD '60-100')
        ages = ["60-100", "60-100"]
    elif k == 3:                         # turns 14 during follow-up -> stays child (13)
        ages = ["13", "14"]
    else:
        a0 = rep_ages[k % 9]
        ages = [f"{a0}", f"{a0 + 1}.0"]  # mixed '12' / '13.0' formatting as in v3
    for t, a in zip(["2020-01-01 00:00:00", "2021-01-01 00:00:00"], ages):
        rows_d.append((t, pid, 120.0, a, "F", db))
demo = pd.DataFrame(rows_d, columns=["ts", "PtID", "GlucoseCGM", "Age", "Sex", "Database"])
demo.to_csv(out / "demographics.csv", index=False)
print(f"wrote {len(main)} rows to {out}")
