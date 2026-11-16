"""Figure: risk-adjusted uplift by decile of predicted effect (test set).

Reads results/supplementary/uplift_deciles_test.csv and qini_summary_test.csv;
draws no new estimates. Run from the repository root:  python -m scripts.plot_deciles
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "results/supplementary/uplift_deciles_test.csv"
OUT = ROOT / "results/figures/uplift_deciles_test.png"
SERIES = [("Two-stage model, Logistic", "Two-stage model (logistic)", "#2a78d6"),
          ("Stage 1 alone, Logistic", "Stage 1 alone (logistic)", "#eb6834")]
INK, MUTED, GRID = "#1d1d1f", "#5f5f66", "#e4e4e8"
# Adjusted uplift across all test letters (equals always-one minus always-four).
OVERALL = float(pd.read_csv(ROOT / "results/supplementary/qini_summary_test.csv")["All letters adjusted uplift (pp)"].iloc[0])

d = pd.read_csv(SRC)
fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=200)
x = np.arange(1, 11)
width = 0.38
for k, (key, label, color) in enumerate(SERIES):
    part = d[d["Model"] == key].sort_values("Decile (1 = largest predicted effect)")
    pos = x + (k - 0.5) * (width + 0.02)
    y = part["Adjusted uplift (pp)"].to_numpy()
    err = np.vstack([y - part["95% low (pp)"], part["95% high (pp)"] - y])
    ax.bar(pos, y, width=width, color=color, label=label, zorder=2)
    ax.errorbar(pos, y, yerr=err, fmt="none", ecolor=INK, elinewidth=0.8, capsize=0, alpha=0.55, zorder=3)
ax.axhline(0, color=MUTED, linewidth=0.8, zorder=1)
ax.hlines(OVERALL, 0.4, 10.6, color=INK, linewidth=1, linestyles=(0, (4, 3)), zorder=1)
ax.text(10.7, OVERALL, f"All letters\n{OVERALL:.2f}", va="center", ha="left", fontsize=8, color=INK, linespacing=1.1)
ax.set_xticks(x, [str(i) for i in x])
ax.set_xlabel("Decile of predicted effect of the one-example letter (1 = largest)", fontsize=9, color=INK)
ax.set_ylabel("Adjusted uplift in loans\n(percentage points)", fontsize=9, color=INK)
ax.tick_params(colors=MUTED, labelsize=8)
ax.yaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color(GRID)
ax.set_xlim(0.4, 11.9)
ax.legend(frameon=False, fontsize=8, loc="upper right", ncol=2, bbox_to_anchor=(1.0, 1.12))
fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, facecolor="white")
print("Saved", OUT)
