"""Standard learners built with scikit-learn and XGBoost."""
import warnings

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from .data import APPLICATION, FEATURES, LOAN, TREATMENT

STAGES = ("Application", "Conditional borrowing", "Direct borrowing")
SEEDS = dict(zip(STAGES, [20261102, 20261107, 20261108]))


def interactions(x):
    """Append format-by-covariate products; the format column is last."""
    x = np.asarray(x)
    return np.column_stack([x, x[:, :-1] * x[:, -1:]])


def build_model(method, features, parameters, seed):
    logistic = method == "Logistic S"
    parts = [
        ("categories", OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first" if logistic else None), ["risk", "wave"]),
        ("numeric", StandardScaler() if logistic else "passthrough", [c for c in features if c not in ["risk", "wave"]]),
    ]
    if method != "XGB T":
        parts.append(("letter", "passthrough", [TREATMENT]))
    steps = [("prepare", ColumnTransformer(parts))]
    if logistic:
        steps.append(("interactions", FunctionTransformer(interactions)))
        estimator = LogisticRegression(max_iter=2000, **parameters)
    else:
        estimator = XGBClassifier(objective="binary:logistic", eval_metric="logloss", n_estimators=150, learning_rate=.05,
                                  subsample=1., colsample_bytree=1., random_state=seed, n_jobs=2, **parameters)
    return Pipeline(steps + [("model", estimator)])


def selected_parameters(stage, feature_set, method):
    if method == "Logistic S":
        return {"C": 10. if stage == "Direct borrowing" and feature_set == "baseline" else 1.}
    if stage == "Application":
        depth, child = (2, 20) if (feature_set, method) == ("history", "XGB S") else (1, 50)
    elif stage == "Conditional borrowing":
        depth, child = 1, 20
    else:
        depth, child = 2, 50
    return dict(max_depth=depth, min_child_weight=child, reg_lambda=10)


def fit_one(frame, stage, feature_set, method, parameters):
    """Fit one model; preprocessing is learned from `frame` only."""
    features = FEATURES[feature_set]
    subset = frame.loc[frame[APPLICATION].eq(1)] if stage == "Conditional borrowing" else frame
    target = APPLICATION if stage == "Application" else LOAN
    def fit(part, columns):
        model = build_model(method, features, parameters, SEEDS[stage])
        # Treat convergence warnings as errors so a bad fit can't slip through.
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model.fit(part[columns], part[target].astype(int))
        return model
    if method == "XGB T":
        return {arm: fit(subset.loc[subset[TREATMENT].eq(arm)], features) for arm in (0, 1)}
    return fit(subset, features + [TREATMENT])


def fit_selected(train):
    models = {}
    for stage in STAGES:
        models[stage] = {}
        for feature_set in FEATURES:
            methods = ["Logistic S", "XGB S", "XGB T"] if stage == "Application" else ["Logistic S", "XGB S"]
            for method in methods:
                parameters = selected_parameters(stage, feature_set, method)
                models[stage][feature_set, method] = fit_one(train, stage, feature_set, method, parameters)
        print(f"Fitted {stage} models on training records only.", flush=True)
    return models


def predict_one(model, frame, feature_set, method):
    features = FEATURES[feature_set]
    probabilities = np.empty((len(frame), 2))
    for arm in (0, 1):
        if method == "XGB T":
            probabilities[:, arm] = model[arm].predict_proba(frame[features])[:, 1]
        else:
            inputs = frame[features + [TREATMENT]].copy()
            inputs[TREATMENT] = arm
            probabilities[:, arm] = model.predict_proba(inputs)[:, 1]
    if not np.isfinite(probabilities).all() or not ((probabilities >= 0) & (probabilities <= 1)).all():
        raise ValueError("Invalid model probabilities")
    return probabilities


def predict_all(models, frame):
    predictions = {stage: {key: predict_one(model, frame, *key) for key, model in stage_models.items()}
                   for stage, stage_models in models.items()}
    predictions["Two-stage borrowing"] = {}
    for key, r in predictions["Application"].items():
        q_key = (key[0], "Logistic S" if key[1] == "Logistic S" else "XGB S")
        predictions["Two-stage borrowing"][key] = r * predictions["Conditional borrowing"][q_key]
    return predictions
