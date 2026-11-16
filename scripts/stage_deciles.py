"""Decile tables for each stage of the model, on the training and test sets.

For the history logistic and XGBoost S versions, customers are ranked by what
each part of the model predicts, split into ten equal groups, and compared with
what actually happened:

- Stage 1: ranked by predicted application uplift; realized application uplift.
- Stage 2: applicants ranked by predicted conversion; observed conversion rate.
- Combined: ranked by predicted loan uplift; realized loan uplift.

Uplift is adjusted with the same doubly robust scores as the main evaluation.
Each ranking gets a Spearman rank correlation between predicted and realized
decile values, with a bootstrap interval. Training-set results are in-sample
and expected to look better; the test set is the honest check.

Run from the repository root:  python -m scripts.stage_deciles
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.data import APPLICATION, LOAN, TREATMENT, load_data, split_data
from src.evaluation import assignment_probabilities, counterfactual_scores
from src.models import fit_selected, predict_all
from scripts.describe_models import verify_against_saved

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/supplementary"
VERSIONS = {"Logistic S": "Logistic", "XGB S": "XGBoost S"}
REPS = 2000


def decile_ids(score, tie_key=None, seed=20261118):
    """Decile 1 = largest score; ties broken in a fixed random order."""
    if tie_key is None:
        tie_key = np.random.default_rng(seed).random(len(score))
    order = np.lexsort((tie_key, -score))
    ids = np.empty(len(score), dtype=int)
    for d, ix in enumerate(np.array_split(order, 10), start=1):
        ids[ix] = d
    return ids


def summarize(pred, realized, ids):
    """Mean predicted and realized value per decile (both in percentage points)."""
    rows = []
    for d in range(1, 11):
        m = ids == d
        rows.append((d, m.sum(), 100 * pred[m].mean(), 100 * realized[m].mean()))
    return pd.DataFrame(rows, columns=["Decile", "Records", "Predicted (pp)", "Realized (pp)"])


def spearman_with_ci(pred, realized, ids, rng):
    table = summarize(pred, realized, ids)
    rho = spearmanr(table["Predicted (pp)"], table["Realized (pp)"]).statistic
    draws = []
    n = len(pred)
    for _ in range(REPS):
        ix = rng.integers(0, n, n)
        t = summarize(pred[ix], realized[ix], ids[ix])
        draws.append(spearmanr(t["Predicted (pp)"], t["Realized (pp)"]).statistic)
    lo, hi = np.nanquantile(draws, [.025, .975])
    return table, float(rho), float(lo), float(hi)


def analyse(name, frame, train, models, rng):
    pred = predict_all(models, frame)
    t = frame[TREATMENT].to_numpy(dtype=int)
    applied = frame[APPLICATION].to_numpy(dtype=int)
    loan = frame[LOAN].to_numpy(dtype=int)
    e, _, _ = assignment_probabilities(train, frame)
    # Doubly robust uplift scores: applications use the history XGBoost S
    # application model as the outcome model; loans use the direct loan model,
    # exactly as in the main evaluation.
    phi_app = counterfactual_scores(applied, t, e, pred["Application"]["history", "XGB S"])
    phi_loan = counterfactual_scores(loan, t, e, pred["Direct borrowing"]["history", "XGB S"])
    up_app = phi_app[:, 1] - phi_app[:, 0]
    up_loan = phi_loan[:, 1] - phi_loan[:, 0]

    # On the test set, break ties in the combined ranking exactly as
    # targeting_metrics.py does, so the deciles match Figure 3 in the paper.
    fig3_keys = {}
    if name == "Test":
        key_rng = np.random.default_rng(20261116)
        fig3_keys = {"Logistic S": key_rng.random(len(frame)), "XGB S": key_rng.random(len(frame))}

    deciles, tests = [], []
    for method, label in VERSIONS.items():
        r = pred["Application"]["history", method]
        q = pred["Conditional borrowing"]["history", method]
        b = pred["Two-stage borrowing"]["history", method]
        mask = applied == 1
        q_obs = q[np.arange(len(frame)), t]           # predicted conversion under the letter received
        parts = [
            ("Stage 1: application uplift", r[:, 1] - r[:, 0], up_app, np.ones(len(frame), bool)),
            ("Stage 2: conversion among applicants", q_obs, loan.astype(float), mask),
            ("Combined: loan uplift", b[:, 1] - b[:, 0], up_loan, np.ones(len(frame), bool)),
        ]
        for part, p, y, m in parts:
            key = fig3_keys.get(method) if part.startswith("Combined") else None
            ids = decile_ids(p[m], tie_key=key)
            table, rho, lo, hi = spearman_with_ci(p[m], y[m], ids, rng)
            overall = 100 * y[m].mean()
            table.insert(0, "Part", part); table.insert(0, "Model", label); table.insert(0, "Data set", name)
            table["Ratio to overall"] = table["Realized (pp)"] / overall
            deciles.append(table)
            top = table.loc[table["Decile"] == 1, "Realized (pp)"].item()
            tests.append({"Data set": name, "Model": label, "Part": part, "Records": int(m.sum()),
                          "Overall (pp)": overall, "Top decile (pp)": top, "Top-decile ratio": top / overall,
                          "Spearman rho": rho, "95% low": lo, "95% high": hi})
    return deciles, tests


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    parts = split_data(load_data())
    train, test = parts["training"], parts["test"]
    models = fit_selected(train)
    diff, _ = verify_against_saved(train, test, models)
    print(f"Refitted models match the original test scores (max difference {diff:.1e}).")
    rng = np.random.default_rng(np.random.SeedSequence(20261118).spawn(1)[0])
    all_deciles, all_tests = [], []
    for name, frame in [("Training", train), ("Test", test)]:
        d, s = analyse(name, frame, train, models, rng)
        all_deciles += d; all_tests += s
    pd.concat(all_deciles).to_csv(OUT / "stage_deciles.csv", index=False)
    tests = pd.DataFrame(all_tests)
    tests.to_csv(OUT / "stage_rank_order_tests.csv", index=False)
    print(tests.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
