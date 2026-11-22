"""Prompt 7 - small 1D CNN on the raw evening CGM sequence (72 readings).

Same data rules as the baselines:
  * same subject split and same nights (taken from features_<tag>_{split}.csv,
    so the evening-coverage filter and any --exclude-low-in-progress match)
  * tuned / early-stopped on VALIDATION only; test is used once at the end
  * same threshold rule (highest threshold reaching --target-sens on val)
  * same metrics and the same subject-level bootstrap (same seed -> same draws)

Input to the network, per night: 2 channels x 72 time steps
  ch 0: glucose, missing values linearly interpolated (edges: nearest value),
        then standardised with the TRAIN-set mean and SD only
  ch 1: 1 where the reading was missing, else 0 (so the model knows what was filled)

Network (about 6k weights - deliberately small for this data size):
  Conv(2->16,k5)-BN-ReLU-MaxPool2 -> Conv(16->32,k5)-BN-ReLU-MaxPool2
  -> Conv(32->32,k3)-ReLU -> global average pool -> Dropout 0.3 -> Linear(32->1)
Training: Adam lr 1e-3, weight decay 1e-4, batch 256, up to 100 epochs,
  loss = BCE with pos_weight = (#neg / #pos in train)  [class weighting],
  early stopping: stop after 10 epochs without a better validation AUROC and
  restore the best epoch's weights.
Seeds 0-4: each seed retrains from scratch.

Reported
  results/cnn_results_seeds.csv   per seed x eval group metrics, + mean and SD rows
  results/cnn_results.csv         the 5-seed AVERAGE prediction, with subject
                                  bootstrap 95% CIs (same format as baseline_results.csv)
  results/cnn_vs_baselines.csv    paired bootstrap difference CNN - each global
                                  baseline (AUROC, AUPRC) per age group, + verdict
  results/cnn_test_predictions_<tag>.csv, cnn_validation_predictions_<tag>.csv,
  cnn_val_scores_<tag>.csv (ensemble AUROC is used for model selection),
  results/cnn_training_log.csv

Usage
  pip install torch            # CPU version is fine
  python src/cnn.py
  (run baselines.py first - the comparison reads its test predictions)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import diadata_io as io
from baselines import (METRICS, SubjectBootstrap, ci, metrics,
                       threshold_for_sensitivity)

PID = "PtID"
SEEDS = (0, 1, 2, 3, 4)


# ---------------- data ----------------
def interpolate_rows(ev: np.ndarray) -> np.ndarray:
    """Linear interpolation of NaNs within each row; edges take nearest value."""
    out = ev.astype("float32").copy()
    t = np.arange(ev.shape[1])
    for i in range(len(out)):
        ok = ~np.isnan(out[i])
        if ok.all():
            continue
        if not ok.any():
            raise ValueError("an evening window has no readings at all")
        out[i, ~ok] = np.interp(t[~ok], t[ok], out[i, ok])
    return out


def build_inputs(ev_by_split: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """[n,72] glucose with NaN -> [n,2,72] float32, scaled with TRAIN stats only."""
    filled = {s: interpolate_rows(e) for s, e in ev_by_split.items()}
    mu = float(np.nanmean(ev_by_split["train"]))
    sd = float(np.nanstd(ev_by_split["train"]))
    return {s: np.stack([(filled[s] - mu) / sd,
                         np.isnan(ev_by_split[s]).astype("float32")], axis=1).astype("float32")
            for s in ev_by_split}


def load(tag: str, results: Path, processed: Path, exclude_lip: bool, exclude_source: str | None):
    win = np.load(processed / f"windows_{tag}.npz")
    pos = pd.Series(np.arange(len(win["night_id"])), index=win["night_id"])
    tables, ev = {}, {}
    for s in ("train", "val", "test"):
        d = pd.read_csv(results / "features" / f"features_{tag}_{s}.csv", dtype={PID: str})
        if exclude_lip:
            d = d[d["low_in_progress_at_start"] == 0]
        if exclude_source:
            drop = [x.strip() for x in exclude_source.split(",") if x.strip()]
            d = d[~d["source_dataset"].isin(drop)]
        tables[s] = d.reset_index(drop=True)
        ev[s] = win["evening"][pos.loc[d["night_id"]].to_numpy()]
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        assert not set(tables[a][PID]) & set(tables[b][PID]), f"subject leak {a}/{b}"
    return tables, ev


# ---------------- model (PyTorch imported only here) ----------------
def fit_predict(seed, X, y, args):
    """Train one CNN. Returns (val_probs, test_probs, log_rows)."""
    import random

    import torch
    from torch import nn

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)                    # removes thread-order randomness on CPU

    net = nn.Sequential(
        nn.Conv1d(2, 16, 5, padding=2), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2),
        nn.Conv1d(16, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
        nn.Conv1d(32, 32, 3, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(32, 1))
    pos_weight = torch.tensor([(y["train"] == 0).sum() / max((y["train"] == 1).sum(), 1)],
                              dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)

    Xt = {s: torch.from_numpy(X[s]) for s in X}
    yt = torch.from_numpy(y["train"].astype("float32"))
    gen = torch.Generator().manual_seed(seed)

    def predict(xs):
        net.eval()
        with torch.no_grad():
            return torch.sigmoid(net(xs).squeeze(1)).numpy()

    best_auc, best_state, bad, log = -np.inf, None, 0, []
    for epoch in range(args.max_epochs):
        net.train()
        order = torch.randperm(len(yt), generator=gen)
        total = 0.0
        for k in range(0, len(order), args.batch):
            idx = order[k:k + args.batch]
            if len(idx) < 2:                   # BatchNorm needs >1 sample
                continue
            opt.zero_grad()
            loss = loss_fn(net(Xt["train"][idx]).squeeze(1), yt[idx])
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(idx)
        val_auc = roc_auc_score(y["val"], predict(Xt["val"]))
        log.append({"seed": seed, "epoch": epoch, "train_loss": total / len(yt),
                    "val_auroc": val_auc})
        if val_auc > best_auc + 1e-4:
            best_auc, bad = val_auc, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    net.load_state_dict(best_state)
    print(f"  seed {seed}: best val AUROC {best_auc:.3f} at epoch "
          f"{max(r['epoch'] for r in log) - bad}")
    return predict(Xt["val"]), predict(Xt["test"]), log


# ---------------- evaluation ----------------
def evaluate(tables, y, val_p, test_p, args, fit_log):
    test = tables["test"]
    groups = sorted(test["age_group"].unique())
    eval_sets = {"All": np.arange(len(test))}
    eval_sets.update({g: np.flatnonzero(test["age_group"].to_numpy() == g) for g in groups})

    # per-seed metrics (training randomness)
    seed_rows = []
    for s in SEEDS:
        thr = threshold_for_sensitivity(y["val"], val_p[s], args.target_sens)
        for g, ix in eval_sets.items():
            seed_rows.append({"seed": s, "eval_group": g, **metrics(y["test"][ix], test_p[s][ix], thr)})
    seeds_df = pd.DataFrame(seed_rows)
    agg = seeds_df.groupby("eval_group")[METRICS].agg(["mean", "std"])
    agg.columns = [f"{m}_{a}" for m, a in agg.columns]
    seeds_out = pd.concat([seeds_df, agg.reset_index().assign(seed="mean/sd")], ignore_index=True)

    # 5-seed average prediction + subject bootstrap CIs
    pv = np.mean([val_p[s] for s in SEEDS], axis=0)
    pt = np.mean([test_p[s] for s in SEEDS], axis=0)
    thr = threshold_for_sensitivity(y["val"], pv, args.target_sens)

    base = None
    bp = args.results / args.baseline_predictions
    if bp.exists():
        base = pd.read_csv(bp, dtype={PID: str})
        base = base[base["scope"] == "global"]
    else:
        print(f"NOTE: {bp} not found - skipping CNN vs baseline comparison.")

    rows, comp = [], []
    for g, ix in eval_sets.items():
        sub = test.iloc[ix]
        bs = SubjectBootstrap(sub[PID].to_numpy(), args.n_boot, args.seed)
        idx_list = list(bs.indices())
        yy, pp = y["test"][ix], pt[ix]
        point = metrics(yy, pp, thr)
        boots = [metrics(yy[b], pp[b], thr) for b in idx_list]
        row = {"model": "cnn_5seed_avg", "trained_on": "global", "eval_group": g,
               "n_subjects": len(bs.uniq), "n_nights": len(yy), "n_positive": int(yy.sum()),
               "prevalence": round(yy.mean(), 4), "threshold": round(thr, 4)}
        for k in METRICS:
            row[k] = point[k]
            row[f"{k}_ci_low"], row[f"{k}_ci_high"] = ci([b[k] for b in boots])
            row[f"{k}_seed_mean"] = agg.loc[g, f"{k}_mean"]
            row[f"{k}_seed_sd"] = agg.loc[g, f"{k}_std"]
        rows.append(row)

        if base is None:
            continue
        for bm, bdf in base.groupby("model"):
            bdf = bdf.set_index("night_id")
            if not sub["night_id"].isin(bdf.index).all():
                print(f"NOTE: baseline {bm} lacks some test nights for {g}; skipped.")
                continue
            bpp = bdf.loc[sub["night_id"], "prob"].to_numpy()
            bthr = float(bdf["threshold"].iloc[0])
            bpoint = metrics(yy, bpp, bthr)
            for k in ("auroc", "auprc"):
                d = [metrics(yy[b], pp[b], thr)[k] - metrics(yy[b], bpp[b], bthr)[k]
                     for b in idx_list]
                lo, hi = ci(d)
                verdict = ("CNN better" if lo > 0 else "CNN worse" if hi < 0
                           else "no clear difference")
                comp.append({"eval_group": g, "baseline": f"{bm} (global)", "metric": k,
                             "cnn": point[k], "baseline_value": bpoint[k],
                             "cnn_minus_baseline": point[k] - bpoint[k],
                             "ci_low": lo, "ci_high": hi, "verdict": verdict})
    ensemble_val_auc = roc_auc_score(y["val"], pv)
    return seeds_out, pd.DataFrame(rows), pd.DataFrame(comp), pv, pt, thr, ensemble_val_auc


def main(argv=None, fit=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default=io.PRIMARY_TAG)
    ap.add_argument("--seed", type=int, default=42, help="bootstrap seed (match baselines)")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--target-sens", type=float, default=0.80)
    ap.add_argument("--max-epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--exclude-low-in-progress", action="store_true")
    ap.add_argument("--exclude-source", default=None,
                    help="drop source dataset(s) from every split, comma-separated")
    ap.add_argument("--exclude-label", default=None,
                    help="short name used in output file names when dropping a source")
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
    args = ap.parse_args(argv)
    suffix = (f"_{args.tag}" + ("_noLIP" if args.exclude_low_in_progress else "")
              + (f"_no-{args.exclude_label or args.exclude_source.replace(',', '+')}"
                 if args.exclude_source else ""))
    args.baseline_predictions = f"baseline_test_predictions{suffix}.csv"
    fit = fit or fit_predict

    tables, ev = load(args.tag, args.results, args.processed,
                      args.exclude_low_in_progress, args.exclude_source)
    X = build_inputs(ev)
    y = {s: tables[s]["label"].to_numpy() for s in tables}
    print(f"nights: train {len(y['train'])}, val {len(y['val'])}, test {len(y['test'])}; "
          f"train positives {y['train'].mean():.1%}")

    val_p, test_p, logs = {}, {}, []
    for s in SEEDS:
        vp, tp, log = fit(s, X, y, args)
        val_p[s], test_p[s] = vp, tp
        logs += log

    seeds_out, res, comp, pv, pt, thr, ensemble_val_auc = evaluate(
        tables, y, val_p, test_p, args, logs)
    tag_out = "" if suffix == f"_{io.PRIMARY_TAG}" else suffix
    seeds_out.round(4).to_csv(args.results / f"cnn_results_seeds{tag_out}.csv", index=False)
    res.round(4).to_csv(args.results / f"cnn_results{tag_out}.csv", index=False)
    comp.round(4).to_csv(args.results / f"cnn_vs_baselines{tag_out}.csv", index=False)
    pd.DataFrame(logs).to_csv(args.results / f"cnn_training_log{tag_out}.csv", index=False)
    pd.DataFrame({"night_id": tables["test"]["night_id"], PID: tables["test"][PID],
                  "age_group": tables["test"]["age_group"], "label": y["test"],
                  "model": "cnn_5seed_avg", "scope": "global", "prob": pt,
                  "threshold": thr, "pred": (pt >= thr).astype(int)}
                 ).to_csv(args.results / f"cnn_test_predictions{suffix}.csv", index=False)
    pd.DataFrame({"night_id": tables["val"]["night_id"], PID: tables["val"][PID],
                  "age_group": tables["val"]["age_group"], "label": y["val"],
                  "model": "cnn_5seed_avg", "scope": "global", "prob": pv}
                 ).to_csv(args.results / f"cnn_validation_predictions{suffix}.csv", index=False)
    # Keep individual best-epoch scores for transparency, and record the actual
    # five-seed ensemble score used for model selection.
    L = pd.DataFrame(logs)
    if "val_auroc" in L:
        val_scores = L.groupby("seed")["val_auroc"].max().rename("best_val_auroc").reset_index()
        val_scores["score_type"] = "seed_best_epoch_descriptive"
        val_scores = pd.concat([val_scores, pd.DataFrame([{
            "seed": "ensemble", "best_val_auroc": ensemble_val_auc,
            "score_type": "five_seed_average_used_for_selection"}])], ignore_index=True)
        val_scores.to_csv(args.results / f"cnn_val_scores{suffix}.csv", index=False)

    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print("\nCNN (5-seed average) with bootstrap CIs, and mean ± SD over seeds:")
        print(res[["eval_group", "n_nights", "prevalence", "auroc", "auroc_ci_low",
                   "auroc_ci_high", "auroc_seed_mean", "auroc_seed_sd", "auprc",
                   "sensitivity", "specificity", "false_alarms_per_10_nights"]]
              .round(3).to_string(index=False))
        if len(comp):
            print("\nCNN minus baseline (paired subject bootstrap):")
            print(comp.round(3).to_string(index=False))
            n_better = (comp["verdict"] == "CNN better").sum()
            print(f"\nHONEST SUMMARY: the CNN is clearly better in {n_better} of {len(comp)} "
                  f"comparisons; '{'no clear difference'}' or worse in the rest. "
                  "Only claim an advantage where the CI excludes 0.")


if __name__ == "__main__":
    main()
