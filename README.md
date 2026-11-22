# Predicting Clock-Defined Overnight Hypoglycemia Across Age Groups in DiaData

This folder contains analysis code and synthetic-data test generators for the EMBC 2027 manuscript by Kavin Pandiarajan. Get the DiaData inputs separately. Keep the downloaded data, participant-level predictions and generated outputs local.

## Set up Python

The analysis used Python 3.11 and the exact package versions in `requirements-lock.txt`. Run the commands below from the repository root.

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
```

Optional checks without DiaData:

```sh
python tests/run_checks.py
python tests/check_cnn_no_torch.py
bash tests/run_e2e.sh .venv/bin/python
```

The tests generate their own synthetic inputs. The end-to-end test covers the baseline, error analysis, figures and robustness pipeline. The mocked CNN test checks the CNN pipeline outside model training.

## Download DiaData v3

1. Open the [DiaData v3 Zenodo deposit](https://doi.org/10.5281/zenodo.17285631) and download `5_min_sampling.zip`. The release is by B. Cinar and M. Maleshkova under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). See the [dataset paper](https://doi.org/10.1051/bioconf/202519503001) for the source studies.
2. Open `5_min_sampling/raw.zip` inside the download. Extract `MDB_5min_sampling_raw.csv` and `SDBI_5min_sampling_raw.csv` into `data/raw/`. The two extracted files total about 14.2 GB. Allow additional space for the archive, temporary files and results.
3. Check the CSV hashes from the repository root. The analysis reads the raw 5-min release and uses the per-reading `Age` in SDBI. The standalone `Demographics.csv` and the quality-enhanced releases are separate files.

```sh
shasum -a 256 data/raw/MDB_5min_sampling_raw.csv data/raw/SDBI_5min_sampling_raw.csv
```

| Raw file | Expected SHA-256 |
|---|---|
| `MDB_5min_sampling_raw.csv` | `69a68df24ebe6ebcf19b1b7cd586751b87fbb9abb53825ab62414483b63c8a40` |
| `SDBI_5min_sampling_raw.csv` | `b2b15ce6fa1eca00e094628a036b1965890f24d0826cad99ede8c90880f6f173` |

## Run the primary analysis

```sh
diadata_main=data/raw/MDB_5min_sampling_raw.csv
diadata_age=data/raw/SDBI_5min_sampling_raw.csv
embc_tag=n00-06_lt70_r4

python src/explore_data.py --main "$diadata_main" --demographics "$diadata_age"
python src/nights.py --main "$diadata_main" --demographics "$diadata_age"
python src/features.py --tag "$embc_tag" --seed 42
python src/source_composition.py --tag "$embc_tag"
python src/baselines.py --tag "$embc_tag" --seed 42 --n-boot 1000
python src/cnn.py --tag "$embc_tag" --seed 42 --n-boot 1000
python src/error_analysis.py --tag "$embc_tag" --seed 42 --n-boot 1000
python src/figures.py --tag "$embc_tag" --fig1-model hgb --factor last_glucose
```

Preserve `results/splits_seed42.csv` throughout the run. The primary definition uses an onset from 00:00 up to but excluding 06:00 on the dataset clock, with at least four successive observed 5-min slots below 70 mg/dL. Predictors use only slots before midnight. The age-stratified 70/15/15 split is at the subject-ID level. The CNN averages seeds 0–4. The error analysis selects the global model by validation AUROC and evaluates the threshold set on validation.

## Run the sensitivity analyses

```sh
python src/within_source.py --tag "$embc_tag" --model hgb --seed 42 --n-boot 1000
python src/age_at_night_sensitivity.py --demographics "$diadata_age"
python src/robustness.py --main "$diadata_main" --demographics "$diadata_age" --model hgb --seed 42 --n-boot 1000
python src/features.py --tag "$embc_tag" --variant-suffix _cov70 --min-evening-coverage 0.70 --seed 42
python src/baselines.py --tag "${embc_tag}_cov70" --seed 42 --n-boot 1000
python src/error_analysis.py --tag "${embc_tag}_cov70" --model hgb --seed 42 --n-boot 1000
```

The RT-CGM and age-at-night checks reuse the primary HGB scores. The Table II variants reuse the subject split and retune HGB using validation data. The 70% coverage run produces separate files with the `_cov70` suffix.

### Reproduce the post hoc PEDAP exclusion

Run this block after the primary analysis. It matches the saved global HGB scores to the test features and excludes PEDAP at evaluation. The existing bootstrap function draws 1,000 subject resamples separately within each age group. Sorting by subject ID reproduces the diagnostic's bootstrap draw order.

```sh
PYTHONPATH=src python - <<'PY'
from pathlib import Path

import pandas as pd

from error_analysis import primary_age_gap

results = Path("results")
tag = "n00-06_lt70_r4"
keys = ["night_id", "PtID", "age_group", "label"]
features = pd.read_csv(
    results / "features" / f"features_{tag}_test.csv",
    usecols=keys + ["source_dataset"],
    dtype={"PtID": str},
)
scores = pd.read_csv(results / f"baseline_test_predictions_{tag}.csv", dtype={"PtID": str})
global_hgb = scores.loc[(scores["model"] == "hgb") & (scores["scope"] == "global")]
test = features.merge(global_hgb[keys + ["prob"]], on=keys, validate="one_to_one")
assert len(test) == len(features) == len(global_hgb)
without_pedap = test.loc[test["source_dataset"] != "PEDAP"].sort_values("PtID", kind="stable")
result = primary_age_gap(without_pedap, n_boot=1000, seed=42)
point, low, high = (result[key] for key in ("children_minus_adults", "ci_low", "ci_high"))
print(f"{point:.4f} (nominal 95% CI {low:.4f} to {high:.4f})")
PY
```

Expected output is `-0.0111 (nominal 95% CI -0.0505 to 0.0355)`.

## Expected outputs

| Output path | Contents |
|---|---|
| `results/raw_checksums.txt`, `results/subject_summary.csv` | Input hashes and age assignments |
| `data/processed/windows_n00-06_lt70_r4.npz`, `results/nights_n00-06_lt70_r4.csv` | Aligned evening and overnight windows and labels |
| `results/features/`, `results/source_composition_n00-06_lt70_r4.csv` | Split feature tables, exclusions and age-by-source counts |
| `results/baseline_results.csv`, `results/cnn_results.csv`, `results/cnn_vs_baselines.csv` | Test performance and model comparisons. Validation and prediction CSVs are saved alongside them. |
| `results/error_analysis/n00-06_lt70_r4/` | Primary AUROC contrast, locked-threshold error analysis and feature comparisons |
| `results/within_source_n00-06_lt70_r4_hgb.csv`, `results/age_at_night_sensitivity/` | Within-source contrast and age-at-night sensitivity |
| `results/robustness_summary.csv`, `results/error_analysis/n00-06_lt70_r4_cov70/` | Sensitivity runs and the separate 70% evening-coverage results |
| `results/figures/` | Vector PDFs and PNGs, including `fig1_auroc_by_age` and `fig2_last_glucose_missed_vs_caught` |

Expected primary checks are 203,619 eligible nights from 1,554 subjects, 28,371 test nights from 230 subjects, global HGB test AUROC 0.745 and child-minus-adult AUROC −0.0397 (95% CI −0.0799 to 0.0006). The primary contrast CSV is printed to five decimal places. The manuscript's CI was rounded directly from full-precision bootstrap endpoints.
