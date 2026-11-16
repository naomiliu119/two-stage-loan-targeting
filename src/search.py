"""Rerun the original tuning search on the training data.

This is for checking the search. run_analysis.py always uses the settings
that were selected originally.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import ParameterGrid

from .data import APPLICATION, FEATURES, LOAN, TREATMENT, training_folds
from .models import STAGES, fit_one, predict_one


def grids(stage):
    logistic = [{"C": c} for c in [.01, .1, 1., 10.]]
    if stage == "Conditional borrowing":
        grid = {"max_depth": [1, 2], "min_child_weight": [10, 20], "reg_lambda": [10]}
    else:
        grid = {"max_depth": [1, 2, 3], "min_child_weight": [20, 50], "reg_lambda": [1, 10]}
    xgb = sorted(ParameterGrid(grid), key=lambda p: (p["max_depth"], -p["min_child_weight"], -p["reg_lambda"]))
    return {"XGB S": xgb, **({"XGB T": xgb} if stage == "Application" else {}), "Logistic S": logistic}


def reproduce_search(train):
    folds = training_folds(train)
    tables = {}
    for stage in STAGES:
        y = train[APPLICATION if stage == "Application" else LOAN].to_numpy(dtype=int)
        t = train[TREATMENT].to_numpy(dtype=int)
        eligible = train[APPLICATION].eq(1).to_numpy() if stage == "Conditional borrowing" else np.ones(len(train), bool)
        rows = []
        jobs = [(features, method, params) for features in FEATURES for method, grid in grids(stage).items() for params in grid]
        for candidate, (features, method, params) in enumerate(jobs):
            actual = np.full(len(train), np.nan)
            fold_losses = []
            for fit, check in folds:
                fit_frame = train.iloc[fit]
                check = check[eligible[check]]
                model = fit_one(fit_frame, stage, features, method, params)
                p = predict_one(model, train.iloc[check], features, method)
                actual[check] = p[np.arange(len(check)), t[check]]
                fold_losses.append(log_loss(y[check], actual[check], labels=[0, 1]))
            if not np.isfinite(actual[eligible]).all():
                raise ValueError("Missing OOF predictions")
            rows.append({"Candidate": candidate, "Feature set": features, "Method": method,
                         "CV log loss": log_loss(y[eligible], actual[eligible], labels=[0, 1]),
                         "CV Brier": brier_score_loss(y[eligible], actual[eligible]),
                         "Fold log loss SD": np.std(fold_losses, ddof=1), "Iteration limit fits": 0,
                         "Parameters": params, "Fold log losses": fold_losses})
            print(f"{stage}: {candidate+1}/{len(jobs)} configurations", flush=True)
        tables[stage] = rows
    return tables
