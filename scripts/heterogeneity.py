"""Does the effect of the one-example letter differ across customers?

Estimates the effect of the letter on loans within groups of customers
(risk category, mailing round, offered rate, previous loans), using the same
doubly robust scores as the main evaluation. For each grouping it also tests
whether the group effects are equal, with a chi-square test on the group means.

Run from the repository root:  python -m scripts.heterogeneity
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.data import LOAN, TREATMENT, load_data, split_data
from src.evaluation import assignment_probabilities, counterfactual_scores
from src.models import fit_selected, predict_all
from scripts.describe_models import verify_against_saved

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/supplementary"


def groupings(frame, train):
    """Customer groups. Rate and loan-history cut points come from the training set."""
    rate_cuts = np.quantile(train["offer4"], [1 / 3, 2 / 3])
    rate = pd.cut(frame["offer4"], [-np.inf, *rate_cuts, np.inf],
                  labels=[f"Low (≤ {rate_cuts[0]:.2f}%)", f"Middle", f"High (> {rate_cuts[1]:.2f}%)"])
    loans = pd.cut(frame["trcount"], [0, 1, 4, np.inf], labels=["1", "2–4", "5 or more"])
    return {
        "Risk category": frame["risk"].str.title(),
        "Mailing round": "Wave " + frame["wave"].astype(int).astype(str),
        "Offered interest rate": rate,
        "Previous loans": loans,
    }


def group_effects(uplift, labels, name):
    rows = []
    df = pd.DataFrame({"u": uplift, "g": labels.to_numpy()})
    for g, part in df.groupby("g", observed=True, sort=False):
        n = len(part)
        mean = part.u.mean()
        se = part.u.std(ddof=1) / np.sqrt(n)
        rows.append({"Grouping": name, "Group": str(g), "Letters": n,
                     "Effect (loans per 1,000)": 1000 * mean,
                     "95% low": 1000 * (mean - 1.96 * se), "95% high": 1000 * (mean + 1.96 * se),
                     "_mean": mean, "_se": se})
    table = pd.DataFrame(rows)
    w = 1 / table["_se"] ** 2
    pooled = np.sum(w * table["_mean"]) / np.sum(w)
    chi2 = float(np.sum(w * (table["_mean"] - pooled) ** 2))
    dof = len(table) - 1
    test = {"Grouping": name, "Groups": len(table), "Chi-square": chi2, "df": dof,
            "p-value": float(stats.chi2.sf(chi2, dof))}
    return table.drop(columns=["_mean", "_se"]), test


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    parts = split_data(load_data())
    train, test = parts["training"], parts["test"]
    models = fit_selected(train)
    diff, _ = verify_against_saved(train, test, models)
    print(f"Refitted models match the original test scores (max difference {diff:.1e}).")

    pred = predict_all(models, test)
    y, t = test[LOAN].to_numpy(dtype=int), test[TREATMENT].to_numpy(dtype=int)
    e, _, _ = assignment_probabilities(train, test)
    phi = counterfactual_scores(y, t, e, pred["Direct borrowing"]["history", "XGB S"])
    uplift = phi[:, 1] - phi[:, 0]
    print(f"Overall effect on the test set: {1000 * uplift.mean():.2f} loans per 1,000 letters.")

    tables, tests = [], []
    for name, labels in groupings(test, train).items():
        table, result = group_effects(uplift, labels, name)
        tables.append(table)
        tests.append(result)
    pd.concat(tables).to_csv(OUT / "heterogeneity_by_group_test.csv", index=False)
    pd.DataFrame(tests).to_csv(OUT / "heterogeneity_tests_test.csv", index=False)
    print(pd.concat(tables).round(2).to_string(index=False))
    print(pd.DataFrame(tests).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
