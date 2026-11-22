"""Prompt 9 - three figures for a 2-column IEEE paper (3.5 in wide, 300 dpi, PDF).

  fig1_auroc_by_age.pdf       AUROC + 95% CI per age group, global vs age-specific
                              (for the baseline model type with the best validation
                              AUROC, or --fig1-model)
  fig2_<factor>_missed_vs_caught.pdf
                              subject-level median of an input feature for missed
                              vs caught positive test nights, by age group
  fig3_onset_time_by_age.pdf  onset time of nocturnal lows per age group (all
                              subjects, all splits - this is descriptive, not a model
                              result), one small panel per group, % of that group's
                              positive nights per 30-min bin

Colours: blue #2a78d6 / orange #eb6834 (checked for colour-blind separation),
and every series also differs by marker or hatch so it survives grey-scale print.

Usage
  python src/figures.py --tag n00-06_lt70_r4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from error_analysis import FACTORS, lower_age  # noqa: E402

BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#222222", "#888888"
W = 3.5  # inches, one IEEE column
AGE_LABELS = {
    "child 0-13": "Child 0–13",
    "teen 14-17": "Adolescent 14–17",
    "adult 18+": "Adult ≥18",
}
AGE_TICK_LABELS = {
    "child 0-13": "Child\n0–13",
    "teen 14-17": "Adolescent\n14–17",
    "adult 18+": "Adult\n≥18",
}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
    "xtick.color": INK, "ytick.color": INK, "axes.labelcolor": INK,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "pdf.fonttype": 42, "ps.fonttype": 42,      # embed fonts (IEEE PDF eXpress)
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def save(fig, path: Path):
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))       # preview copy
    plt.close(fig)
    print(f"saved {path}")


def fig1(res: pd.DataFrame, model: str, out: Path):
    r = res[(res.model == model) & (res.eval_group != "All")]
    groups = sorted(r.eval_group.unique(), key=lower_age)
    x = np.arange(len(groups))
    fig, ax = plt.subplots(figsize=(W, 2.2))
    for dx, trained, col, mk, lab in [(-0.12, "global", BLUE, "o", "Global model"),
                                      (0.12, "age-specific", ORANGE, "s", "Age-specific model")]:
        d = r[r.trained_on == trained].set_index("eval_group").reindex(groups)
        ax.errorbar(x + dx, d.auroc, yerr=[d.auroc - d.auroc_ci_low, d.auroc_ci_high - d.auroc],
                    fmt=mk, ms=4.5, color=col, ecolor=col, elinewidth=1, capsize=2, label=lab)
    ax.set_xticks(x, [AGE_TICK_LABELS.get(g, g) for g in groups])
    ax.set_xlabel("Age group (years)")
    ax.set_ylabel("Test AUROC (95% CI)")
    # Keep the chance-level reference at the lower edge so small AUROC gaps
    # are not visually amplified by a data-dependent truncated axis.
    ax.set_ylim(0.5, 1.0)
    ax.grid(axis="y", color="#e6e6e6", lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
              handletextpad=0.3, columnspacing=1.2)
    print(f"Fig 1 model: {model} - say this in the caption")
    save(fig, out / "fig1_auroc_by_age.pdf")


def fig2(nf: pd.DataFrame, factor: str, out: Path):
    groups = sorted(nf.age_group.unique(), key=lower_age)
    fig, ax = plt.subplots(figsize=(W, 2.3))
    x = np.arange(len(groups))
    for dx, st, col, hatch in [(-0.18, "caught", BLUE, None), (0.18, "missed", ORANGE, "////")]:
        data = [nf.loc[(nf.age_group == g) & (nf.status == st), factor].dropna().to_numpy()
                for g in groups]
        bp = ax.boxplot(data, positions=x + dx, widths=0.3, patch_artist=True,
                        showfliers=False, medianprops=dict(color=INK, lw=1),
                        whiskerprops=dict(color=MUTED, lw=0.7), capprops=dict(color=MUTED, lw=0.7))
        for b, d in zip(bp["boxes"], data):
            b.set(facecolor=col, alpha=0.85, edgecolor="white", lw=0.8, hatch=hatch)
            if len(d) < 5:          # too few nights for a box: hide it, show the points
                b.set_visible(False)
        for i, d in enumerate(data):
            if 0 < len(d) < 5:
                ax.scatter(np.full(len(d), x[i] + dx), d, s=10, color=col, zorder=3,
                           edgecolor=INK, lw=0.3)
                for key in ("medians", "whiskers", "caps"):
                    for art in bp[key][(2 * i if key != "medians" else i):
                                       (2 * i + 2 if key != "medians" else i + 1)]:
                        art.set_visible(False)
    counts = [f"{AGE_TICK_LABELS.get(g, g)}\n({((nf.age_group == g) & (nf.status == 'caught')).sum()} / "
              f"{((nf.age_group == g) & (nf.status == 'missed')).sum()})" for g in groups]
    ax.set_xticks(x, counts)
    ax.set_xlabel("Age group, years (subjects with caught / missed nights)")
    ax.set_ylabel(FACTORS.get(factor, factor), fontsize=7)
    ax.set_title("Each value is a subject-level median", fontsize=7, loc="left", pad=30)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=BLUE, label="Caught (true positive)"),
                       Patch(facecolor=ORANGE, hatch="////", edgecolor="white",
                             label="Missed (false negative)")],
              frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=2)
    ax.grid(axis="y", color="#e6e6e6", lw=0.5)
    ax.set_axisbelow(True)
    save(fig, out / f"fig2_{factor}_missed_vs_caught.pdf")


def fig3(nights: pd.DataFrame, night_start: int, out: Path):
    pos = nights[(nights.label == 1) & (nights.age_group != "unknown")]
    groups = sorted(pos.age_group.unique(), key=lower_age)
    length = int(np.ceil((pos.onset_min_after_start.max() + 1) / 30) * 30)
    bins = np.arange(0, length + 30, 30)
    fig, axes = plt.subplots(len(groups), 1, figsize=(W, 0.85 * len(groups) + 0.5),
                             sharex=True, sharey=True, gridspec_kw={"hspace": 0.55})
    axes = np.atleast_1d(axes)
    for ax, g in zip(axes, groups):
        v = pos.loc[pos.age_group == g, "onset_min_after_start"].to_numpy()
        h, _ = np.histogram(v, bins=bins)
        ax.bar(bins[:-1], 100 * h / max(len(v), 1), width=28, align="edge",
               color=BLUE, edgecolor="white", lw=0.5)
        ax.set_title(
            f"{AGE_LABELS.get(g, g)} (n={len(v)} nights)",
            loc="left", fontsize=7, color=INK, pad=2,
        )
        ax.grid(axis="y", color="#e6e6e6", lw=0.5)
        ax.set_axisbelow(True)
    ticks = bins[::2]
    axes[-1].set_xticks(ticks, [f"{(night_start * 60 + t) // 60 % 24:02d}:{(t % 60):02d}"
                                for t in ticks])
    axes[-1].set_xlabel("Clock time of low onset (30-min bins)")
    fig.supylabel("% of the group's low nights", fontsize=8)
    save(fig, out / "fig3_onset_time_by_age.pdf")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="n00-06_lt70_r4")
    ap.add_argument("--fig1-model", choices=["auto", "logreg", "hgb"], default="auto")
    ap.add_argument("--factor", default=None, choices=list(FACTORS))
    ap.add_argument("--results", type=Path, default=Path("results"))
    args = ap.parse_args()
    out = args.results / "figures"
    out.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}"

    res_path = args.results / ("baseline_results.csv" if args.tag == "n00-06_lt70_r4"
                               else f"baseline_results{suffix}.csv")
    res = pd.read_csv(res_path)
    model = args.fig1_model
    if model == "auto":
        v = pd.read_csv(args.results / f"baseline_val_scores{suffix}.csv")
        v = v[v.scope == "global"]
        model = v.loc[v.filter(like="val_").iloc[:, 0].idxmax(), "model"]
    fig1(res, model, out)

    ea = args.results / "error_analysis" / args.tag
    chosen = json.loads((ea / "chosen.json").read_text())
    fig2(pd.read_csv(ea / "subject_factors_by_status.csv"), args.factor or chosen["key_factor"], out)

    nights = pd.read_csv(args.results / f"nights_{args.tag}.csv")
    fig3(nights, int(args.tag[1:3]), out)


if __name__ == "__main__":
    main()
