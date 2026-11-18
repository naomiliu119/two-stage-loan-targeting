"""Appendix figures B1, B2, D1 and D2, drawn in the same style as the paper.

B1: how much worse each tuning candidate did than the best one (two-stage model).
B2: application-model log loss by tree depth (XGBoost).
D1/D2: calibration of the history logistic and XGBoost S versions on the validation set.

B1 and B2 read the stored cross-validation results; D1 and D2 refit the selected
models. No model is selected or changed here.

Run from the repository root:  python -m scripts.plot_appendix
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import load_data, split_data
from src.evaluation import calibration_bins
from src.models import fit_selected, predict_all

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/figures"
INK, MUTED, GRID = "#1d1d1f", "#5f5f66", "#e4e4e8"
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#8a8a93"
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix",
                     "axes.edgecolor": GRID, "xtick.color": MUTED, "ytick.color": MUTED})
METHOD_LABEL = {"Logistic S": "Logistic S", "XGB S": "XGBoost S", "XGB T": "XGBoost T"}
METHOD_COLOR = {"Logistic S": GREY, "XGB S": BLUE, "XGB T": ORANGE}


def tidy(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def candidates():
    stored = json.loads((ROOT / "results/reference/training_cv_candidates.json").read_text())["stages"]
    rows = []
    for stage in ("Application", "Conditional borrowing"):
        for c in stored[stage]:
            rows.append({"Stage": stage, "Set": c["Feature set"], "Method": c["Method"],
                         "Loss": c["CV log loss"], **c["Parameters"]})
    d = pd.DataFrame(rows)
    d["Excess"] = 1000 * (d["Loss"] - d.groupby(["Stage", "Set", "Method"])["Loss"].transform("min"))
    return d


def figure_b1(d):
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.6), dpi=200)
    titles = {"Application": "Stage 1: application", "Conditional borrowing": "Stage 2: borrowing among applicants"}
    rng = np.random.default_rng(1)
    for ax, stage in zip(axes, titles):
        part = d[d.Stage == stage]
        groups = [(s, m) for s in ("baseline", "history") for m in ("Logistic S", "XGB S", "XGB T")
                  if not part[(part.Set == s) & (part.Method == m)].empty]
        for k, (s, m) in enumerate(groups):
            g = part[(part.Set == s) & (part.Method == m)]
            x = k + rng.uniform(-0.18, 0.18, len(g))
            ax.scatter(x, g.Excess, s=16, color=METHOD_COLOR[m], alpha=0.85, zorder=3)
            ax.scatter(k, 0, s=46, marker="D", facecolor="white", edgecolor=INK, linewidth=1, zorder=4)
        ax.axhline(0, color=MUTED, linewidth=0.8)
        ax.set_xticks(range(len(groups)), [f"{METHOD_LABEL[m]}\n{s.title()}" for s, m in groups], fontsize=9)
        ax.set_title(titles[stage], fontsize=11, color=INK)
        ax.set_ylabel("Log loss above the best\nsetting (×1,000)", fontsize=9.5, color=INK)
        tidy(ax)
    fig.tight_layout()
    fig.savefig(OUT / "training_cv_candidate_spread.png", facecolor="white")
    plt.close(fig)


def figure_b2(d):
    part = d[(d.Stage == "Application") & d.Method.isin(["XGB S", "XGB T"])]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), dpi=200, sharey=True)
    for ax, (s, m) in zip(axes.flat, [(s, m) for s in ("baseline", "history") for m in ("XGB S", "XGB T")]):
        g = part[(part.Set == s) & (part.Method == m)]
        color = METHOD_COLOR[m]
        ax.scatter(g.max_depth + np.linspace(-0.08, 0.08, len(g)) % 0.16 - 0.08, g.Loss, s=16, color=color, alpha=0.8, zorder=3)
        best = g.groupby("max_depth")["Loss"].min()
        ax.plot(best.index, best.values, color=color, linewidth=1.2, zorder=2)
        pick = g.loc[g.Loss.idxmin()]
        ax.scatter(pick.max_depth, pick.Loss, s=110, marker="*", color=INK, zorder=4)
        ax.set_title(f"{METHOD_LABEL[m]}, {s} predictors", fontsize=10.5, color=INK)
        ax.set_xticks([1, 2, 3])
        ax.set_xlabel("Maximum tree depth", fontsize=9.5, color=INK)
        tidy(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel("Cross-validated log loss", fontsize=9.5, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "application_cv_depth_profile.png", facecolor="white")
    plt.close(fig)


def figure_d(bins, method, name):
    panels = [("Application", "Stage 1: applied"), ("Conditional borrowing", "Stage 2: borrowed, given application"),
              ("Two-stage borrowing", "Combined: took up a loan")]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), dpi=200)
    for ax, (stage, title) in zip(axes, panels):
        g = bins[(bins.Approach == stage) & (bins.Model == method) & (bins["Feature set"] == "history")].sort_values("Bin")
        x, y = g["Mean predicted (%)"].to_numpy(), g["Observed (%)"].to_numpy()
        err = np.maximum(0, [y - g["Wilson low (%)"].to_numpy(), g["Wilson high (%)"].to_numpy() - y])
        low, high = min(x.min(), (y - err[0]).min()), max(x.max(), (y + err[1]).max())
        pad = 0.08 * (high - low)
        lim = (low - pad, high + pad)
        ax.plot(lim, lim, linestyle=(0, (4, 3)), color=MUTED, linewidth=0.9)
        ax.errorbar(x, y, yerr=err, fmt="o-", color=BLUE, markersize=4, linewidth=1.2, elinewidth=0.8, capsize=2)
        ax.set(xlim=lim, ylim=lim)
        ax.set_title(title, fontsize=9.5, color=INK)
        ax.set_xlabel("Mean predicted (%)", fontsize=9, color=INK)
        ax.tick_params(labelsize=8)
        tidy(ax)
    axes[0].set_ylabel("Observed (%)", fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / name, facecolor="white")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = candidates()
    figure_b1(d)
    figure_b2(d)
    parts = split_data(load_data())
    models = fit_selected(parts["training"])
    bins = calibration_bins(parts["validation"], predict_all(models, parts["validation"]))
    figure_d(bins, "Logistic S", "development_calibration_logistic_intervals.png")
    figure_d(bins, "XGB S", "development_calibration_xgb_intervals.png")
    print("Saved Figures B1, B2, D1 and D2 to", OUT)


if __name__ == "__main__":
    main()
