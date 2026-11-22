"""FAKE raw CGM data with realistic-ish nocturnal lows, to test the whole pipeline
end to end (Prompts 3-10). Not real data. Seed 0.

* 160 subjects, 25 days each, 5-min readings, 4 source datasets (A largest)
* Age groups cycle through DiaData's fine codes 0-8
* Probability of a nocturnal low rises as bedtime glucose falls; children get a
  noisier link (so their lows should be harder to predict)
* A low = fall over 45 min to a nadir of 45-66 mg/dL, held 30 min, then recovery
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "fake_rich_raw"
out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)
n_subj, n_days = 160, 25
t0 = pd.Timestamp("2021-03-01")
slots = np.arange(n_days * 288)
hour = (slots % 288) / 12
broad = {0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 2, 7: 2, 8: 3}
parts, demo = [], []
for k in range(n_subj):
    code = k % 9
    child = code <= 3
    src = "A" if k % 2 == 0 else "BCD"[k % 3]
    mean = rng.normal(150, 15)
    g = (mean + 35 * np.sin(2 * np.pi * (hour - 14) / 24)
         + 25 * np.exp(-((hour - 8.5) ** 2) / 1.5) + 30 * np.exp(-((hour - 19.5) ** 2) / 2))
    noise = np.zeros(len(slots))
    sd = 6 if child else 3
    for i in range(1, len(slots)):                     # AR(1) wander
        noise[i] = 0.97 * noise[i - 1] + rng.normal(0, sd)
    g = g + noise
    for d in range(1, n_days):
        mid = d * 288
        bedtime = g[mid - 1]
        link_noise = rng.normal(0, 2.0 if child else 0.7)
        p = 1 / (1 + np.exp(0.035 * (bedtime - 115) + 1.2 - link_noise))
        if rng.random() < p:
            onset = mid + rng.integers(6, 60)              # 00:30-05:00
            nadir = rng.uniform(45, 66)
            fall = np.linspace(g[onset - 9], nadir, 9)
            g[onset - 9:onset] = fall
            g[onset:onset + 6] = nadir + rng.normal(0, 1.5, 6)
            g[onset + 6:onset + 18] = np.linspace(nadir, g[onset + 18], 12)
    g = np.clip(g, 40, 400)
    keep = rng.random(len(slots)) > (0.12 if child else 0.06)
    ts = t0 + pd.to_timedelta(slots[keep] * 5, unit="min")
    parts.append(pd.DataFrame({"ts": ts.strftime("%Y-%m-%d %H:%M:%S"), "PtID": f"F{k:03d}",
                               "GlucoseCGM": np.round(g[keep], 1), "Database": src}))
    demo.append({"PtID": f"F{k:03d}", "Age": str(3 + code * 7) if k % 5 else "10-14",
                 "AgeGroup": code, "AgeGroupBroad": broad[code], "Sex": "M"})
pd.concat(parts).to_csv(out / "maindatabase.csv", index=False)
pd.DataFrame(demo).to_csv(out / "demographics.csv", index=False)
print(f"wrote {sum(len(p) for p in parts)} rows to {out}")
