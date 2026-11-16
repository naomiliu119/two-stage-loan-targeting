"""Describe the selected models: their coefficients, what drives the predicted
effect of the letter, and a train-versus-test check for overfitting.

The models are refitted with their selected settings, and the script first
checks that they reproduce the original test scores, so everything here
describes the same models as the paper.

Run from the repository root:  python -m scripts.describe_models
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from src.data import APPLICATION, FEATURES, LOAN, TREATMENT, load_data, split_data
from src.evaluation import probability_scores
from src.models import fit_selected, predict_all, predict_one, selected_parameters

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/supplementary"
NAMES = {"risk": "Risk category", "offer4": "Offered interest rate", "wave": "Mailing round",
         "trcount": "Previous loans", TREATMENT: "Letter format"}


def verify_against_saved(train, test, models):
    """Check that the refitted models reproduce the original test scores."""
    fresh = probability_scores(train, test, predict_all(models, test)).set_index(["Approach", "Feature set", "Model"])
    reference = json.loads((ROOT / "results/reference/original_final_aggregates.json").read_text())
    saved = pd.DataFrame(reference["probability_scores"]).set_index(["Approach", "Feature set", "Model"])
    joined = fresh[["Log loss", "Brier"]].join(saved[["Log loss", "Brier"]], rsuffix=" saved", how="inner")
    if len(joined) != len(saved):
        raise AssertionError("Some saved scores have no matching refitted model.")
    diff = float(np.abs(joined[["Log loss", "Brier"]].to_numpy() - joined[["Log loss saved", "Brier saved"]].to_numpy()).max())
    if diff > 2e-6:
        raise AssertionError(f"Refitted models differ from the saved scores by {diff}.")
    return diff, saved


def column_names(pipeline):
    """Readable names for the transformed columns (before interactions)."""
    names = []
    for name, transformer, cols in pipeline.named_steps["prepare"].transformers_:
        if name == "categories":
            names += [f"{c}={v}" for c, values in zip(cols, transformer.categories_)
                      for v in (values[1:] if transformer.drop is not None else values)]
        elif name != "remainder":
            names += list(cols)
    return names


def base_variable(column):
    return column.split("=")[0]


def logistic_coefficients(models):
    rows = []
    for stage in ["Application", "Conditional borrowing"]:
        pipe = models[stage]["history", "Logistic S"]
        names = column_names(pipe)  # covariates..., oneln_trt (last)
        coef = pipe.named_steps["model"].coef_.ravel()
        k = len(names)
        terms = names + [f"{n} × one-example letter" for n in names[:-1]]
        rows.append({"Stage": stage, "Term": "Intercept", "Coefficient": float(pipe.named_steps["model"].intercept_[0])})
        for term, value in zip(terms, coef):
            rows.append({"Stage": stage, "Term": term.replace(TREATMENT, "One-example letter"), "Coefficient": float(value)})
        if len(coef) != 2 * k - 1:
            raise AssertionError("Unexpected logistic design width.")
    table = pd.DataFrame(rows)
    table["Odds ratio"] = np.exp(table["Coefficient"])
    return table


def transformed(pipe, frame, features, arm):
    x = frame[features + [TREATMENT]].copy()
    x[TREATMENT] = arm
    return pipe.named_steps["prepare"].transform(x)


def logistic_attribution(pipe, train, frame, features):
    """Exact linear SHAP per arm (interventional, training means as baseline),
    mapped to probability with p(1-p), then differenced across arms."""
    names = column_names(pipe)[:-1]
    model = pipe.named_steps["model"]
    beta = model.coef_.ravel()
    k = len(names)
    b, gamma, delta = beta[:k], beta[k], beta[k + 1:]
    mean = pipe.named_steps["prepare"].transform(train[features + [TREATMENT]])[:, :k].mean(axis=0)
    out = {}
    for arm in (0, 1):
        z = transformed(pipe, frame, features, arm)[:, :k]
        p = pipe.predict_proba(frame[features + [TREATMENT]].assign(**{TREATMENT: arm}))[:, 1]
        phi = (b + delta * arm) * (z - mean)
        out[arm] = (phi * (p * (1 - p))[:, None])
    uplift = out[1] - out[0]
    return pd.DataFrame(uplift, columns=names)


def xgb_attribution(pipe, frame, features):
    import shap  # only needed for the XGBoost attribution
    names = column_names(pipe)
    explainer = shap.TreeExplainer(pipe.named_steps["model"], feature_perturbation="tree_path_dependent")
    out = {}
    for arm in (0, 1):
        z = transformed(pipe, frame, features, arm)
        phi = explainer.shap_values(z)  # margin (log-odds) space
        p = pipe.predict_proba(frame[features + [TREATMENT]].assign(**{TREATMENT: arm}))[:, 1]
        out[arm] = phi * (p * (1 - p))[:, None]
    uplift = pd.DataFrame(out[1] - out[0], columns=names)
    return uplift.drop(columns=[TREATMENT])


def summarize_attribution(uplift, actual_uplift, label):
    grouped = uplift.T.groupby(base_variable).sum().T * 100  # percentage points
    rows = [{"Model": label, "Variable": NAMES.get(v, v),
             "Mean |contribution| (pp)": grouped[v].abs().mean(),
             "Mean signed contribution (pp)": grouped[v].mean(),
             "SD of contribution (pp)": grouped[v].std()} for v in grouped.columns]
    table = pd.DataFrame(rows).sort_values("Mean |contribution| (pp)", ascending=False)
    table["Rank"] = range(1, len(table) + 1)
    summary = {"Model": label, "Mean predicted uplift (pp)": 100 * float(np.mean(actual_uplift)),
               "SD predicted uplift (pp)": 100 * float(np.std(actual_uplift)),
               "Share recommending one-example (%)": 100 * float(np.mean(actual_uplift > 1e-10))}
    return table, summary


def overfitting(train, validation, test_scores, models):
    cv = json.loads((ROOT / "results/reference/training_cv_candidates.json").read_text())["stages"]
    rows = []
    for stage, approach in [("Application", "Application"), ("Conditional borrowing", "Conditional borrowing")]:
        target = APPLICATION if stage == "Application" else LOAN
        for (fs, method), model in models[stage].items():
            def score(frame):
                f = frame.loc[frame[APPLICATION].eq(1)] if stage == "Conditional borrowing" else frame
                probs = predict_one(model, f, fs, method)
                t = f[TREATMENT].to_numpy(dtype=int)
                return log_loss(f[target].astype(int), probs[np.arange(len(f)), t], labels=[0, 1])
            params = selected_parameters(stage, fs, method)
            cand = [c for c in cv[stage] if c["Feature set"] == fs and c["Method"] == method and c["Parameters"] == params]
            if len(cand) != 1:
                raise AssertionError(f"Cannot match CV row for {stage} {fs} {method}")
            test_ll = float(test_scores.loc[(approach, fs, method), "Log loss"])
            train_ll = score(train)
            rows.append({"Stage": stage, "Feature set": fs, "Model": method,
                         "Training (in-sample)": train_ll, "Training CV": cand[0]["CV log loss"],
                         "Validation": score(validation), "Test (reported)": test_ll,
                         "Test / training ratio": test_ll / train_ll})
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    parts = split_data(load_data())
    train, validation, test = parts["training"], parts["validation"], parts["test"]
    models = fit_selected(train)
    diff, saved = verify_against_saved(train, test, models)
    print(f"Refitted models match the original test scores (max difference {diff:.1e}).")

    logistic_coefficients(models).to_csv(OUT / "logistic_coefficients_history.csv", index=False)

    features = FEATURES["history"]
    tables, summaries = [], []
    pred_val = predict_all(models, validation)
    for stage, method, label in [("Application", "Logistic S", "Stage 1 · history logistic"),
                                 ("Application", "XGB S", "Stage 1 · history XGBoost S"),
                                 ("Conditional borrowing", "Logistic S", "Stage 2 · history logistic")]:
        pipe = models[stage]["history", method]
        frame = validation if stage == "Application" else validation.loc[validation[APPLICATION].eq(1)]
        up = logistic_attribution(pipe, train, frame, features) if method == "Logistic S" else xgb_attribution(pipe, frame, features)
        probs = pred_val[stage]["history", method]
        if stage == "Conditional borrowing":
            probs = probs[validation[APPLICATION].eq(1).to_numpy()]
        table, summary = summarize_attribution(up, probs[:, 1] - probs[:, 0], label)
        tables.append(table)
        summaries.append(summary)
    q = pred_val["Conditional borrowing"]["history", "XGB S"]
    summaries.append({"Model": "Stage 2 · history XGBoost S", "Mean predicted uplift (pp)": 100 * float(np.mean(q[:, 1] - q[:, 0])),
                      "SD predicted uplift (pp)": 100 * float(np.std(q[:, 1] - q[:, 0])),
                      "Share recommending one-example (%)": float("nan")})
    pd.concat(tables).to_csv(OUT / "uplift_attribution_validation.csv", index=False)
    pd.DataFrame(summaries).to_csv(OUT / "uplift_summary_validation.csv", index=False)

    overfitting(train, validation, saved, models).to_csv(OUT / "overfitting_check.csv", index=False)
    print("Wrote", *sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
