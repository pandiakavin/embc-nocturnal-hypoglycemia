"""Prompt 4 - cut each subject's CGM into nights, label them, save evening inputs.

Default definitions (all are command-line options, so Prompt 10 can change them):
  night     00:00-06:00 clock time (72 five-minute slots), named by its date
  keep      >= 80% of the night's 72 slots have a glucose value
  positive  a confirmed run of >= 4 consecutive readings <70 mg/dL whose
            first reading (event onset) falls in [00:00, 06:00). Readings
            through 06:10 may confirm an onset at 05:55. A run already under
            way at 00:00 is flagged and excluded from the primary feature set.
            Missing readings break observed runs; a negative is flagged
            label_ambiguous_gap if missing values could conceal a qualifying run,
            including a fully unobserved four-slot run.
            All settings live in diadata_io.PRIMARY.
  evening   the 6 h before the night starts (18:00-24:00 the previous day,
            72 slots) - this is the model input. Missing readings stay NaN.

Usage:
  python src/nights.py --main data/raw/maindatabase.csv \
      --demographics data/raw/subdatabaseI.csv --age-scheme broad
  # robustness (Prompt 10):  --night-start 22 --night-end 7
  #                          --threshold 54 --op lt
  #                          --min-run 3 --op le   (earlier 3-reading <=70 rule)
  #                          --interp-max-missing 4 (fill <=4 missing slots,
  #                                                   real readings <30 min apart)

Outputs (the tag in the file name records the settings, e.g. n00-06_lt70_r4):
  results/nights_<tag>.csv            one row per KEPT night (labels + metadata)
  results/nights_summary_<tag>.csv    nights and % positive per age group
  data/processed/windows_<tag>.npz    evening [n,72] and night [n,L] glucose
                                      arrays, row-aligned with the CSV (night_id)
  results/hourly_glucose_by_database.csv + figures/check_hourly_glucose.png
                                      sanity check that timestamps are local time
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

import diadata_io as io

H = io.SLOTS_PER_HOUR   # 12 slots per hour
D = io.SLOTS_PER_DAY    # 288 slots per day


def is_low(x: np.ndarray, thr: float, op: str) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return (x <= thr) if op == "le" else (x < thr)   # NaN -> False


def run_starts(low: np.ndarray, k: int) -> np.ndarray:
    """Boolean [n, L-k+1]: True at t if readings t..t+k-1 are all low."""
    if low.shape[1] < k:
        return np.zeros((low.shape[0], 0), bool)
    return sliding_window_view(low, k, axis=1).all(axis=-1)


def count_episodes(w: np.ndarray) -> np.ndarray:
    """Number of separate qualifying low episodes per row (rising edges of w)."""
    if w.shape[1] == 0:
        return np.zeros(w.shape[0], int)
    rising = w.copy()
    rising[:, 1:] &= ~w[:, :-1]
    return rising.sum(axis=1)


def interp_short_gaps(arr: np.ndarray, max_missing: int) -> np.ndarray:
    """Linearly fill runs of <= max_missing NaNs that have a real reading on BOTH
    sides. Longer gaps are left fully NaN (not partly filled)."""
    out = arr.copy()
    isn = np.isnan(arr)
    if max_missing <= 0 or not isn.any():
        return out
    d = np.diff(np.r_[0, isn.astype(int), 0])
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)   # [start, end)
    for a, b in zip(starts, ends):
        if a == 0 or b == len(arr) or (b - a) > max_missing:
            continue
        out[a:b] = np.interp(np.arange(a, b), [a - 1, b], [arr[a - 1], arr[b]])
    return out


def count_episodes_recovery(low: np.ndarray, observed: np.ndarray, min_run: int,
                            recovery: int) -> np.ndarray:
    """Episodes per row. A run of >= min_run consecutive low readings (a missing
    reading breaks a run, as for the label) starts an episode. The episode ENDS
    only after >= recovery OBSERVED non-low readings (missing readings are not
    evidence of recovery and are not counted). Qualifying runs separated by
    fewer observed non-low readings are merged. Counted where the episode starts."""
    n, L = low.shape
    counts = np.zeros(n, int)
    if n == 0:
        return counts
    pad = np.zeros((n, 1), bool)
    flat_low = np.concatenate([pad, low, pad], axis=1).ravel()
    flat_rec = np.concatenate([pad, observed & ~low, pad], axis=1).ravel()
    d = np.diff(flat_low.astype(np.int8))
    starts = np.flatnonzero(d == 1) + 1        # first low reading (flat index)
    ends = np.flatnonzero(d == -1) + 1         # index after last low reading
    rows = (starts - 1) // (L + 2)
    q = (ends - starts) >= min_run
    starts, ends, rows = starts[q], ends[q], rows[q]
    if len(starts) == 0:
        return counts
    cum_rec = np.r_[0, np.cumsum(flat_rec)]    # cum_rec[i] = recovery readings before i
    rec_between = cum_rec[starts[1:]] - cum_rec[ends[:-1]]
    new_ep = np.ones(len(starts), bool)
    new_ep[1:] = (rows[1:] != rows[:-1]) | (rec_between >= recovery)
    np.add.at(counts, rows[new_ep], 1)
    return counts


def process_subject(slot, glu, args, night_off, night_len, ev_len):
    """Return per-night arrays for one subject (all candidate nights)."""
    u, mean, _ = io.collapse_duplicate_slots(slot, glu)
    first_day, last_day = u[0] // D, u[-1] // D
    base = (first_day - 2) * D                          # 2-day margin each side
    arr = np.full(int((last_day + 3) * D - base), np.nan, dtype="float32")
    arr[u - base] = mean

    days = np.arange(first_day, last_day + 1)           # date of each night's END
    start = days * D + night_off - base                 # index of night's first slot
    N = arr[start[:, None] + np.arange(night_len)]
    # Labels are assigned by event onset. Read min_run-1 slots before the night
    # to detect runs already under way at its start, and min_run-1 slots after
    # the night so an onset before the right boundary can be confirmed.
    pre_label = args.min_run - 1
    post_label = args.min_run - 1
    label_slots = start[:, None] - pre_label + np.arange(
        night_len + pre_label + post_label
    )
    # Label-only copy with short gaps filled in the interpolation sensitivity;
    # features and coverage always use raw readings.
    arr_lab = interp_short_gaps(arr, args.interp_max_missing)
    N_lab = arr_lab[label_slots]
    EV = arr[start[:, None] - ev_len + np.arange(ev_len)]
    P24 = arr[start[:, None] - D + np.arange(D)]        # 24 h before night start
    return days, N, N_lab, EV, P24, u, mean


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--main", type=Path, required=True)
    ap.add_argument("--demographics", type=Path, default=None)
    ap.add_argument("--age-scheme", default="kids_teens_adults", choices=io.AGE_SCHEMES)
    ap.add_argument("--night-start", type=int, default=io.PRIMARY["night_start"], help="hour, 0-23")
    ap.add_argument("--night-end", type=int, default=io.PRIMARY["night_end"], help="hour, 1-23")
    ap.add_argument("--evening-hours", type=int, default=6)
    ap.add_argument("--min-coverage", type=float, default=0.80)
    ap.add_argument("--threshold", type=float, default=io.PRIMARY["threshold"])
    ap.add_argument("--op", choices=["le", "lt"], default=io.PRIMARY["op"], help="le: <=, lt: <")
    ap.add_argument("--min-run", type=int, default=io.PRIMARY["min_run"], help="consecutive low readings")
    ap.add_argument("--recovery", type=int, default=3,
                    help="OBSERVED non-low readings needed to END an episode (3 = 15 min) for the "
                         "prior-24h episode count")
    ap.add_argument("--interp-max-missing", type=int, default=0,
                    help="sensitivity: before LABELLING, linearly fill gaps of up to this many "
                         "missing readings (4 = real readings <30 min apart, as in Eichenlaub 2024). "
                         "0 = off (primary). Features always use raw readings.")
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
    ap.add_argument("--chunksize", type=int, default=2_000_000)
    args = ap.parse_args()

    for d in (args.results, args.processed):
        io.check_output_dir(d, [args.main, args.demographics])
        d.mkdir(parents=True, exist_ok=True)
    (args.results / "figures").mkdir(exist_ok=True)

    # night window in slots, relative to midnight of the night's end date
    s, e = args.night_start, args.night_end
    night_off = s * H if s < e else (s - 24) * H        # e.g. 22:00 -> -2 h
    night_len = ((e - s) % 24) * H
    ev_len = args.evening_hours * H
    tag = io.make_tag(s, e, args.op, args.threshold, args.min_run)
    if args.interp_max_missing:
        tag += f"_i{args.interp_max_missing}"
    print(f"Night {s:02d}:00-{e:02d}:00 ({night_len} slots), evening = {ev_len} slots "
          f"before it, positive = {args.min_run}+ readings {'<=' if args.op == 'le' else '<'}"
          f" {args.threshold:g} mg/dL, keep if coverage >= {args.min_coverage:.0%}")

    main_arr = io.load_main_arrays(args.main, args.chunksize)
    demo = io.load_demographics(args.demographics, args.chunksize)
    labels = io.age_labels(demo, args.age_scheme) if not demo.empty else pd.Series(dtype=str)

    rows, evenings, nights = [], [], []
    n_candidate = {}                                    # nights with any data, per group
    hourly_sum, hourly_n = {}, {}
    for c, a, b in io.split_by_subject(main_arr["code"]):
        pid, src = main_arr["pids"][c], main_arr["db"][c]
        group = labels.get(pid, io.UNKNOWN)
        days, N, N_lab, EV, P24, u, mean = process_subject(
            main_arr["slot"][a:b], main_arr["glu"][a:b], args, night_off, night_len, ev_len)

        # local-time sanity check: mean glucose by clock hour, per source dataset
        ok = ~np.isnan(mean)
        hr = (u[ok] % D) // H
        hourly_sum[src] = hourly_sum.get(src, 0) + np.bincount(hr, mean[ok], minlength=24)
        hourly_n[src] = hourly_n.get(src, 0) + np.bincount(hr, minlength=24)

        cov = (~np.isnan(N)).mean(axis=1)
        n_candidate[group] = n_candidate.get(group, 0) + int((cov > 0).sum())
        keep = cov >= args.min_coverage
        if not keep.any():
            continue
        days, N, N_lab, EV, P24, cov = (days[keep], N[keep], N_lab[keep], EV[keep],
                                        P24[keep], cov[keep])   # coverage uses RAW readings

        lowN = is_low(N_lab, args.threshold, args.op)
        w = run_starts(lowN, args.min_run)
        pre_label = args.min_run - 1
        run_start_minute = np.arange(w.shape[1]) - pre_label
        # Include events whose onset is inside [night start, night end), while
        # allowing the final min_run-1 readings to confirm an onset just before
        # night_end. Also retain a separately flagged event already under way at
        # night_start for the --keep-ongoing sensitivity.
        overlaps_night = ((run_start_minute < night_len) &
                          (run_start_minute + args.min_run > 0))
        w_overlapping = w & overlaps_night[None, :]
        pos = w_overlapping.any(axis=1)
        onset_idx = np.where(
            pos, w_overlapping.argmax(axis=1) - pre_label, -1
        )
        onset_min = np.where(pos, onset_idx * io.SLOT_MIN, np.nan)
        low_in_progress = (w & ((run_start_minute < 0) &
                                (run_start_minute + args.min_run > 0))[None, :]).any(axis=1)

        # A negative is ambiguous when a run of min_run low-or-missing slots
        # could overlap the night and at least one slot is missing. This includes
        # a fully unobserved run; requiring an observed low would incorrectly
        # treat such nights as definite negatives.
        maybe = np.zeros(len(N), bool)
        if w.shape[1]:
            possible = run_starts(lowN | np.isnan(N_lab), args.min_run)
            has_gap = sliding_window_view(np.isnan(N_lab), args.min_run, axis=1).any(axis=-1)
            maybe = (possible & has_gap & overlaps_night[None, :]).any(axis=1)
        p24_eps = count_episodes_recovery(is_low(P24, args.threshold, args.op),
                                          ~np.isnan(P24), args.min_run, args.recovery)

        for i in range(len(days)):
            rows.append({
                io.PID: pid, "source_dataset": src, "age_group": group,
                "night_date": (pd.Timestamp("1970-01-01") + pd.Timedelta(days=int(days[i]))).date(),
                "night_coverage": round(float(cov[i]), 3),
                "evening_coverage": round(float((~np.isnan(EV[i])).mean()), 3),
                "label": int(pos[i]),
                "onset_min_after_start": onset_min[i],
                "onset_clock": (None if not pos[i] else
                                f"{(s * 60 + int(onset_min[i])) // 60 % 24:02d}:"
                                f"{(s * 60 + int(onset_min[i])) % 60:02d}"),
                "night_nadir": float(np.nanmin(N[i])),
        # A qualifying observed low run began before night_start and overlaps
        # the night. The primary feature step excludes it.
                "low_in_progress_at_start": int(low_in_progress[i]),
                # A negative may be unknown if missing data could complete a
                # qualifying low run, including a fully unobserved run.
                "label_ambiguous_gap": int((not pos[i]) and maybe[i]),
                "prior24h_low_episodes": int(p24_eps[i]),
                "prior24h_coverage": round(float((~np.isnan(P24[i])).mean()), 3),
            })
        evenings.append(EV)
        nights.append(N)

    if not rows:
        raise SystemExit("No nights passed the coverage filter - check timestamps/columns.")
    df = pd.DataFrame(rows)
    df.insert(0, "night_id", np.arange(len(df)))
    EV_all = np.concatenate(evenings).astype("float32")
    N_all = np.concatenate(nights).astype("float32")
    df.to_csv(args.results / f"nights_{tag}.csv", index=False)
    np.savez_compressed(args.processed / f"windows_{tag}.npz",
                        night_id=df["night_id"].to_numpy(), evening=EV_all, night=N_all)

    # ---- summary per age group ----
    def summ(g: pd.DataFrame, name: str) -> dict:
        subj_rate = g.groupby(io.PID)["label"].mean()
        ev_ok = g["evening_coverage"] >= args.min_coverage
        return {
            "age_group": name,
            "n_subjects": g[io.PID].nunique(),
            "n_nights_with_any_data": (sum(n_candidate.values()) if name == "All"
                                       else n_candidate.get(name, 0)),
            "n_nights_kept": len(g),
            "n_positive": int(g["label"].sum()),
            "pct_positive": round(100 * g["label"].mean(), 2),
            "n_kept_with_evening_cov_ok": int(ev_ok.sum()),
            "pct_positive_evening_cov_ok": round(100 * g.loc[ev_ok, "label"].mean(), 2) if ev_ok.any() else np.nan,
            "pct_subjects_with_any_positive": round(100 * (subj_rate > 0).mean(), 1),
            "median_subject_pct_positive": round(100 * subj_rate.median(), 2),
            "n_low_in_progress_at_start": int(g["low_in_progress_at_start"].sum()),
            "n_ambiguous_gap_negatives": int(g["label_ambiguous_gap"].sum()),
            "median_onset_min_after_start": g["onset_min_after_start"].median(),
        }
    summary = pd.DataFrame([summ(df, "All")] +
                           [summ(g, k) for k, g in df.groupby("age_group", sort=True)])
    summary.to_csv(args.results / f"nights_summary_{tag}.csv", index=False)

    # ---- local-time check ----
    hourly = pd.DataFrame({src: hourly_sum[src] / np.maximum(hourly_n[src], 1)
                           for src in hourly_sum}, index=pd.RangeIndex(24, name="hour"))
    hourly.round(1).to_csv(args.results / "hourly_glucose_by_database.csv")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    hourly.plot(ax=ax, lw=1)
    ax.set(xlabel="Clock hour in the data", ylabel="Mean glucose (mg/dL)",
           title="Aggregate hourly mean by source (dataset clock; not timezone validation)")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(args.results / "figures" / "check_hourly_glucose.png", dpi=150)

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print("\n", summary.to_string(index=False))
    print(f"\nSaved nights_{tag}.csv, nights_summary_{tag}.csv, windows_{tag}.npz")
    print("Now open results/figures/check_hourly_glucose.png (see README, step 'local time').")


if __name__ == "__main__":
    main()
