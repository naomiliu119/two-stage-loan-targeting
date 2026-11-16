"""Bootstrap intervals for prediction scores, and the refit stability check."""
import numpy as np
import pandas as pd

from .data import APPLICATION, LOAN, TREATMENT
from .models import fit_selected, predict_all


def loss_vectors(y, p):
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    eps = np.finfo(np.float64).eps
    q = np.clip(p, eps, 1-eps)
    return {"Log loss": -(y*np.log(q)+(1-y)*np.log1p(-q)), "Brier": np.square(y-p)}


def prediction_intervals(train, frame, predictions, repetitions=2000):
    """Pointwise paired intervals; conditional scores resample applicants only."""
    t = frame[TREATMENT].to_numpy(dtype=int)
    r, c = (frame[name].to_numpy(dtype=int) for name in [APPLICATION, LOAN])
    constants = {"Application": train[APPLICATION].mean(),
                 "Conditional borrowing": train.loc[train[APPLICATION].eq(1), LOAN].mean(),
                 "Overall borrowing": train[LOAN].mean()}
    series, specs = {}, []
    for stage in ["Application", "Conditional borrowing", "Two-stage borrowing", "Direct borrowing"]:
        cohort = "Applicants" if stage == "Conditional borrowing" else "All records"
        mask = r == 1 if cohort == "Applicants" else np.ones(len(frame), bool)
        y = (r if stage == "Application" else c)[mask]
        task = stage if stage in constants else "Overall borrowing"
        reference = loss_vectors(y, np.full(len(y), constants[task]))
        for (features, method), probabilities in predictions[stage].items():
            losses = loss_vectors(y, probabilities[np.arange(len(frame)), t][mask])
            series[stage, features, method] = losses
            specs.append({"Kind": "Model minus constant", "Approach": stage, "Feature set": features, "Model": method,
                          "Cohort": cohort, "diff": {m: losses[m]-reference[m] for m in losses}})
    for features in ["baseline", "history"]:
        for method in ["Logistic S", "XGB S"]:
            a, b = (series[stage, features, method] for stage in ["Two-stage borrowing", "Direct borrowing"])
            specs.append({"Kind": "Two-stage minus direct", "Approach": "Overall borrowing", "Feature set": features, "Model": method,
                          "Cohort": "All records", "diff": {m: a[m]-b[m] for m in a}})
    results = []
    for cohort, seed in zip(["All records", "Applicants"], np.random.SeedSequence(20261109).spawn(2)):
        columns = [(row, metric) for row in specs if row["Cohort"] == cohort for metric in ["Log loss", "Brier"]]
        matrix = np.column_stack([row["diff"][metric] for row, metric in columns])
        rng = np.random.default_rng(seed)
        draws = np.empty((repetitions, len(columns)))
        for b in range(repetitions):
            ix = rng.integers(0, len(matrix), size=len(matrix))
            draws[b] = matrix[ix].mean(axis=0)
        lows, highs = np.quantile(draws, [.025, .975], axis=0)
        for j, (row, metric) in enumerate(columns):
            results.append({**{k: v for k, v in row.items() if k != "diff"}, "Metric": metric, "Records": len(matrix),
                            "Difference": matrix[:, j].mean(), "Conditional 95% low": lows[j], "Conditional 95% high": highs[j]})
        print(f"Completed {repetitions} paired {cohort.lower()} prediction-score resamples.", flush=True)
    return pd.DataFrame(results)


def refit_stability(train, validation, original_predictions, repetitions=30):
    """Refit the selected models on bootstrap samples of the training data and
    measure how often their recommendations on the validation set stay the same."""
    references = {(stage, *key): 100*(p[:, 1]-p[:, 0]) for stage in ["Application", "Two-stage borrowing", "Direct borrowing"]
                  for key, p in original_predictions[stage].items()}
    contrasts = {key: np.empty((repetitions, len(validation))) for key in references}
    rng = np.random.default_rng(20261104)
    covariates_only = validation.drop(columns=[APPLICATION, LOAN])
    for b in range(repetitions):
        sample = train.iloc[rng.integers(0, len(train), size=len(train))]
        refits = fit_selected(sample)
        predictions = predict_all(refits, covariates_only)
        for (stage, features, method), storage in contrasts.items():
            p = predictions[stage][features, method]
            storage[b] = 100*(p[:, 1]-p[:, 0])
        print(f"Completed {b+1}/{repetitions} matched training refits.", flush=True)
    rows = []
    for (stage, features, method), matrix in contrasts.items():
        original = references[stage, features, method] > 1e-8
        for risk in ["ALL", "HIGH", "LOW", "MEDIUM"]:
            mask = np.ones(len(validation), bool) if risk == "ALL" else validation.risk.eq(risk).to_numpy()
            rows.append({"Approach": stage, "Feature set": features, "Method": method, "Risk": risk,
                         "Validation records": int(mask.sum()),
                         "Original preference retained (%)": 100*((matrix[:, mask] > 1e-8) == original[mask]).mean(),
                         "Average uplift SD (pp)": np.std(matrix[:, mask], axis=0, ddof=1).mean()})
    return pd.DataFrame(rows)
