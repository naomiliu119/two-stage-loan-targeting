"""Static, shareable figures from aggregate analysis outputs."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def calibration_figure(bins, output, method="Logistic S"):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), layout="constrained")
    for ax, stage in zip(axes, ["Application", "Conditional borrowing", "Two-stage borrowing"]):
        for feature_set, color in [("baseline", "#3378a3"), ("history", "#c76634")]:
            part = bins.loc[bins.Approach.eq(stage) & bins.Model.eq(method) & bins["Feature set"].eq(feature_set)]
            if part.empty:
                continue
            y = part["Observed (%)"].to_numpy()
            err = np.maximum(0, np.array([y-part["Wilson low (%)"].to_numpy(), part["Wilson high (%)"].to_numpy()-y]))
            ax.errorbar(part["Mean predicted (%)"], y, yerr=err, fmt="o-", color=color, capsize=3, label=feature_set)
        limits = ax.get_xlim(), ax.get_ylim()
        low, high = min(limits[0][0], limits[1][0]), max(limits[0][1], limits[1][1])
        ax.plot([low, high], [low, high], "--", color="0.6", linewidth=1)
        ax.set(xlim=(low, high), ylim=(low, high), title=stage, xlabel="Mean predicted (%)", ylabel="Observed (%)")
        ax.grid(alpha=.2)
    axes[0].legend(title="Feature set")
    fig.suptitle(f"Development calibration: {method}\nFive quantile bins; pointwise Wilson intervals for observed proportions", fontsize=12)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def policy_figure(matched, output):
    subset = matched.loc[matched["First rule"].str.startswith("Two-stage")].copy()
    labels = subset["First rule"].str.replace("Two-stage | ", "", regex=False)
    y = np.arange(len(subset))
    point = subset["Difference / 1,000"].to_numpy()
    low, high = (subset[c].to_numpy() for c in ["Conditional 95% low", "Conditional 95% high"])
    fig, ax = plt.subplots(figsize=(9, 4.8), layout="constrained")
    ax.errorbar(point, y, xerr=np.maximum(0, [point-low, high-point]), fmt="o", capsize=4, color="#3378a3")
    ax.axvline(0, color="0.4", linestyle="--", linewidth=1)
    ax.set(yticks=y, yticklabels=labels, xlabel="Two-stage minus application-based choices: loans per 1,000", title="Fixed-rule comparisons on the original final-test partition")
    ax.invert_yaxis()
    fig.text(.01, -.035, "Pointwise 95% intervals hold fitted functions fixed. History | Logistic S is the single primary contrast.", fontsize=9)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
