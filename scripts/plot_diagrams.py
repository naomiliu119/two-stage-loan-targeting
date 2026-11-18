"""Schematic figures for the paper (no estimates are computed here).

Figure 1: the letter-to-loan funnel in the analysis sample, with bars drawn to scale.
Figure 2: the structure of the two-stage model and the decision it makes.

Run from the repository root:  python -m scripts.plot_diagrams
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent.parent / "results/figures"
INK, MUTED, LINE = "#1d1d1f", "#5f5f66", "#c9c9cf"
BLUE, BLUE_BG = "#2a78d6", "#e6f0fb"
ORANGE, ORANGE_BG = "#eb6834", "#fdece4"
GREY, GREY_BG = "#8a8a93", "#f3f3f5"
TITLE, BODY, NOTE = 12, 11.5, 10.5
# Serif text and math, to match the paper.
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix"})

# Analysis-sample counts (Section 2).
LETTERS, APPLIED, LOANS = 53_194, 4_522, 3_944


def box(ax, x, y, w, h, face, edge, lw=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12",
                                facecolor=face, edgecolor=edge, linewidth=lw))


def arrow(ax, x0, y0, x1, y1, color=MUTED, lw=1.3, style="-", head=11):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=head,
                                 color=color, linewidth=lw, linestyle=style, shrinkA=0, shrinkB=0))


def funnel():
    """Horizontal bars drawn to scale, starting right after their labels."""
    fig, ax = plt.subplots(figsize=(6.5, 2.2), dpi=200)
    x0, full = 0.0, 1.0              # bar origin and length of the largest bar
    ax.set_xlim(-0.47, 1.15); ax.set_ylim(-0.05, 3.15); ax.axis("off")
    rows = [("Letters mailed", LETTERS, GREY_BG, GREY, 2.55),
            ("Applied (Stage 1)", APPLIED, BLUE_BG, BLUE, 1.4),
            ("Took up a loan (Stage 2)", LOANS, ORANGE_BG, ORANGE, 0.25)]
    h = 0.5
    for label, n, face, edge, y in rows:
        w = full * n / LETTERS
        ax.add_patch(plt.Rectangle((x0, y), w, h, facecolor=face, edgecolor=edge, linewidth=1.2))
        ax.text(x0 - 0.02, y + h / 2, label, ha="right", va="center", fontsize=BODY, color=INK, fontweight="bold")
        ax.text(x0 + w + 0.015, y + h / 2, f"{n:,}", ha="left", va="center", fontsize=BODY, color=INK)
    ax.text(x0, 2.2, f"{100 * (1 - APPLIED / LETTERS):.1f}% of letters led to no application",
            ha="left", va="center", fontsize=NOTE, color=MUTED, style="italic")
    ax.text(x0, 1.05, f"{100 * (1 - LOANS / APPLIED):.1f}% of applicants did not take out a loan",
            ha="left", va="center", fontsize=NOTE, color=MUTED, style="italic")
    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
    fig.savefig(OUT / "funnel_letter_to_loan.png", facecolor="white")
    plt.close(fig)


def model_flow():
    fig, ax = plt.subplots(figsize=(7.2, 4.1), dpi=200)
    ax.set_xlim(0, 12); ax.set_ylim(0.3, 6.0); ax.axis("off")

    # Inputs
    box(ax, 0.05, 2.25, 2.0, 2.1, GREY_BG, LINE)
    ax.text(1.05, 4.0, "Inputs", ha="center", va="top", fontsize=TITLE, fontweight="bold", color=INK)
    ax.text(1.05, 3.3, "Customer\ncharacteristics $x$", ha="center", va="center", fontsize=BODY, color=INK, linespacing=1.25)
    ax.text(1.05, 2.62, "scored under\nboth letters", ha="center", va="center", fontsize=NOTE, color=MUTED, linespacing=1.2)

    # Stage 1 and Stage 2
    box(ax, 2.7, 3.75, 3.6, 2.0, BLUE_BG, BLUE)
    ax.text(4.5, 5.52, "Stage 1: Application model", ha="center", va="top", fontsize=TITLE, fontweight="bold", color=INK)
    ax.text(4.5, 4.75, r"$\hat r_t(x)$ = chance of applying", ha="center", va="center", fontsize=BODY, color=INK)
    ax.text(4.5, 4.12, "fitted on all training letters", ha="center", va="center", fontsize=NOTE, color=MUTED)

    box(ax, 2.7, 0.45, 3.6, 2.0, ORANGE_BG, ORANGE)
    ax.text(4.5, 2.22, "Stage 2: Borrowing model", ha="center", va="top", fontsize=TITLE, fontweight="bold", color=INK)
    ax.text(4.5, 1.45, r"$\hat q_t(x)$ = chance of borrowing," "\n" "given an application", ha="center", va="center", fontsize=BODY, color=INK, linespacing=1.3)
    ax.text(4.5, 0.72, "fitted on training applicants only", ha="center", va="center", fontsize=NOTE, color=MUTED)


    # Combine
    box(ax, 6.85, 2.1, 2.35, 2.4, "white", INK)
    ax.text(8.025, 4.27, "Chance of a loan", ha="center", va="top", fontsize=TITLE, fontweight="bold", color=INK)
    ax.text(8.025, 3.45, r"$\hat r_t(x)\,\hat q_t(x)$", ha="center", va="center", fontsize=13, color=INK)
    ax.text(8.025, 2.68, "under each letter,\n$t$ = 1 and $t$ = 0", ha="center", va="center", fontsize=NOTE, color=MUTED, linespacing=1.3)

    # Decision
    box(ax, 9.7, 2.1, 2.25, 2.4, GREY_BG, LINE)
    ax.text(10.825, 4.27, "Decision", ha="center", va="top", fontsize=TITLE, fontweight="bold", color=INK)
    ax.text(10.825, 3.5, r"$\hat\Delta_C = \hat r_1\hat q_1 - \hat r_0\hat q_0$", ha="center", va="center", fontsize=BODY, color=INK)
    ax.text(10.825, 2.7, r"$> 0$: one example" "\n" r"$\leq 0$: four examples", ha="center", va="center", fontsize=NOTE, color=MUTED, linespacing=1.35)

    # Main flow
    arrow(ax, 2.1, 3.65, 2.65, 4.6)
    arrow(ax, 2.1, 2.95, 2.65, 1.6)
    arrow(ax, 6.35, 4.6, 6.8, 3.95)
    arrow(ax, 6.35, 1.6, 6.8, 2.5)
    arrow(ax, 9.25, 3.3, 9.65, 3.3, color=INK)

    fig.tight_layout(pad=0.2)
    fig.savefig(OUT / "two_stage_model_flow.png", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    funnel()
    model_flow()
    print("Saved funnel_letter_to_loan.png and two_stage_model_flow.png to", OUT)
