"""Schematic figures for the paper (no estimates are computed here).

Figure 1: the letter-to-loan funnel in the analysis sample (counts from Table 1/Section 2).
Figure 2: the structure of the two-stage model (Section 3).
Run from the repository root:  python -m scripts.plot_diagrams
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

OUT = Path(__file__).resolve().parent.parent / "results/figures"
INK, MUTED, LINE = "#1d1d1f", "#5f5f66", "#c9c9cf"
BLUE, BLUE_BG = "#2a78d6", "#e6f0fb"
ORANGE, ORANGE_BG = "#eb6834", "#fdece4"
GREY_BG = "#f3f3f5"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})


def box(ax, x, y, w, h, face, edge, lw=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12",
                                facecolor=face, edgecolor=edge, linewidth=lw))


def arrow(ax, x0, y0, x1, y1, color=MUTED, lw=1.3):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=11,
                                 color=color, linewidth=lw, shrinkA=0, shrinkB=0))


def funnel():
    fig, ax = plt.subplots(figsize=(7.2, 2.6), dpi=200)
    ax.set_xlim(0, 12); ax.set_ylim(0, 4.4); ax.axis("off")
    stages = [("Letters mailed", "53,194", "Mailing rounds 2 and 3", GREY_BG, MUTED, 3.6),
              ("Stage 1: Applied", "4,522", "8.5% of letters", BLUE_BG, BLUE, 2.6),
              ("Stage 2: Took up a loan", "3,944", "87.2% of applicants\n7.4% of letters", ORANGE_BG, ORANGE, 2.3)]
    xs = [0.2, 4.35, 8.3]
    w = 3.4
    for (title, n, sub, face, edge, h), x in zip(stages, xs):
        y = (4.4 - h) / 2
        box(ax, x, y, w, h, face, edge)
        ax.text(x + w / 2, y + h - 0.35, title, ha="center", va="top", fontsize=9, color=INK, fontweight="bold")
        ax.text(x + w / 2, y + h / 2 - 0.05, n, ha="center", va="center", fontsize=16, color=INK)
        if sub:
            ax.text(x + w / 2, y + 0.28, sub, ha="center", va="bottom", fontsize=8, color=MUTED, linespacing=1.15)
    arrow(ax, xs[0] + w + 0.08, 2.2, xs[1] - 0.08, 2.2)
    arrow(ax, xs[1] + w + 0.08, 2.2, xs[2] - 0.08, 2.2)
    fig.tight_layout(pad=0.2)
    fig.savefig(OUT / "funnel_letter_to_loan.png", facecolor="white")
    plt.close(fig)


def model_flow():
    fig, ax = plt.subplots(figsize=(7.2, 3.9), dpi=200)
    ax.set_xlim(0, 12); ax.set_ylim(0, 6.6); ax.axis("off")
    # inputs
    box(ax, 0.05, 2.3, 2.05, 2.0, GREY_BG, LINE)
    ax.text(1.075, 3.95, "Inputs", ha="center", va="top", fontweight="bold", color=INK)
    ax.text(1.075, 3.3, "Customer\ncharacteristics $x$", ha="center", va="center", fontsize=8.5, color=INK, linespacing=1.2)
    ax.text(1.075, 2.65, "Each letter $t$\n(scored twice)", ha="center", va="center", fontsize=8, color=MUTED, linespacing=1.2)
    # stage 1
    box(ax, 2.75, 3.9, 4.25, 2.3, BLUE_BG, BLUE)
    ax.text(4.875, 6.0, "Stage 1: Application model", ha="center", va="top", fontweight="bold", color=INK)
    ax.text(4.875, 5.25, r"$\hat r_t(x)$ = chance of applying", ha="center", va="center", color=INK)
    ax.text(4.875, 4.5, "Fitted on all training letters\nS- or T-learner", ha="center", va="center", fontsize=8, color=MUTED, linespacing=1.2)
    # stage 2
    box(ax, 2.75, 0.4, 4.25, 2.3, ORANGE_BG, ORANGE)
    ax.text(4.875, 2.5, "Stage 2: Borrowing model", ha="center", va="top", fontweight="bold", color=INK)
    ax.text(4.875, 1.75, r"$\hat q_t(x)$ = chance of borrowing if applied", ha="center", va="center", fontsize=8.6, color=INK)
    ax.text(4.875, 1.0, "Fitted on training applicants only\npooled S-learner", ha="center", va="center", fontsize=8, color=MUTED, linespacing=1.2)
    # combine
    box(ax, 7.6, 2.15, 2.0, 2.3, "white", INK, lw=1.2)
    ax.text(8.6, 4.25, "Combine", ha="center", va="top", fontweight="bold", color=INK)
    ax.text(8.6, 3.45, r"$\hat r_t(x)\,\hat q_t(x)$", ha="center", va="center", fontsize=10, color=INK)
    ax.text(8.6, 2.75, "chance of a loan\nunder each letter", ha="center", va="center", fontsize=8, color=MUTED, linespacing=1.2)
    # decision
    box(ax, 10.1, 2.15, 1.85, 2.3, GREY_BG, LINE)
    ax.text(11.025, 4.25, "Decision", ha="center", va="top", fontweight="bold", color=INK)
    ax.text(11.025, 3.4, r"Is $\hat\Delta_C(x) > 0$?", ha="center", va="center", fontsize=8.5, color=INK)
    ax.text(11.025, 2.7, "Yes: one example\nNo: four examples", ha="center", va="center", fontsize=7.5, color=MUTED, linespacing=1.25)
    # arrows
    arrow(ax, 2.15, 3.6, 2.7, 5.0)
    arrow(ax, 2.15, 3.0, 2.7, 1.6)
    arrow(ax, 7.05, 5.0, 7.55, 3.85)
    arrow(ax, 7.05, 1.6, 7.55, 2.75)
    arrow(ax, 9.65, 3.3, 10.05, 3.3, color=INK)
    fig.tight_layout(pad=0.2)
    fig.savefig(OUT / "two_stage_model_flow.png", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    funnel()
    model_flow()
    print("Saved funnel_letter_to_loan.png and two_stage_model_flow.png to", OUT)
