"""Probability scores and the doubly robust estimator used to compare decision rules.

Intervals hold the fitted models fixed. Assignment probabilities are estimated
from the training data, since the original ones were not published.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from .data import APPLICATION, LOAN, TREATMENT


def probability_scores(train, frame, predictions):
    """Evaluate the assigned format, with applicant-only conditional scores."""
    t = frame[TREATMENT].to_numpy(dtype=int)
    rows = []
    def add(stage, features, method, p, y, mask):
        yy, pp = y[mask], np.asarray(p)[mask]
        rows.append({"Approach": stage, "Feature set": features, "Model": method,
                     "Records": len(yy), "Events": int(yy.sum()),
                     "Log loss": log_loss(yy, pp, labels=[0, 1]), "Brier": brier_score_loss(yy, pp),
                     "Observed (%)": 100 * yy.mean(), "Mean predicted (%)": 100 * pp.mean()})
    r, c = (frame[name].to_numpy(dtype=int) for name in [APPLICATION, LOAN])
    all_rows = np.ones(len(frame), dtype=bool)
    for stage, y, mask, constant in [
        ("Application", r, all_rows, train[APPLICATION].mean()),
        ("Conditional borrowing", c, r == 1, train.loc[train[APPLICATION].eq(1), LOAN].mean()),
        ("Overall borrowing", c, all_rows, train[LOAN].mean()),
    ]:
        add(stage, "—", "Original training-average benchmark", np.full(len(frame), constant), y, mask)
    for stage in ["Application", "Conditional borrowing", "Two-stage borrowing", "Direct borrowing"]:
        for (features, method), p in predictions[stage].items():
            add(stage, features, method, p[np.arange(len(frame)), t], r if stage == "Application" else c,
                r == 1 if stage == "Conditional borrowing" else all_rows)
    return pd.DataFrame(rows)


def calibration_bins(frame, predictions):
    """Five development quantile bins; pointwise Wilson observed-rate intervals."""
    rows = []
    t = frame[TREATMENT].to_numpy(dtype=int)
    z = 1.959963984540054
    for stage, models in predictions.items():
        y = frame[APPLICATION if stage == "Application" else LOAN].to_numpy(dtype=int)
        mask = frame[APPLICATION].eq(1).to_numpy() if stage == "Conditional borrowing" else np.ones(len(frame), bool)
        for (features, method), probabilities in models.items():
            p = probabilities[np.arange(len(frame)), t]
            d = pd.DataFrame({"p": p[mask], "y": y[mask]})
            bins = pd.Series(np.zeros(len(d), int)) if d.p.nunique() == 1 else pd.qcut(d.p, 5, labels=False, duplicates="drop")
            for bin_number, part in d.groupby(bins, observed=True):
                n, events = len(part), int(part.y.sum())
                rate = events / n
                den = 1 + z*z/n
                center = (rate + z*z/(2*n))/den
                half = z * np.sqrt(rate*(1-rate)/n + z*z/(4*n*n))/den
                rows.append({"Approach": stage, "Feature set": features, "Model": method, "Bin": int(bin_number)+1,
                             "Records": n, "Events": events, "Mean predicted (%)": 100*part.p.mean(),
                             "Observed (%)": 100*rate, "Wilson low (%)": 100*(center-half), "Wilson high (%)": 100*(center+half)})
    return pd.DataFrame(rows)


def assignment_probabilities(train, frame, keys=("risk", "wave")):
    keys = list(keys)
    counts = train.groupby(keys, observed=True).agg(training_records=(TREATMENT, "size"), one_example_records=(TREATMENT, "sum"))
    counts["estimated_p_one"] = counts.one_example_records / counts.training_records
    positions = counts.index.get_indexer(pd.MultiIndex.from_frame(frame[keys]) if len(keys) > 1 else pd.Index(frame[keys[0]]))
    if (positions < 0).any():
        raise ValueError("A test group does not appear in the training data.")
    e = counts.estimated_p_one.to_numpy()[positions]
    if not np.isfinite(e).all() or not ((e > 0) & (e < 1)).all():
        raise ValueError("An assignment probability is 0 or 1 in some group.")
    return e, counts, positions


def counterfactual_scores(y, t, e, m):
    y, t, e, m = map(np.asarray, (y, t, e, m))
    if y.shape != t.shape or t.shape != e.shape or m.shape != (len(y), 2):
        raise ValueError("Mismatched evaluator inputs")
    if not np.isfinite(e).all() or not ((e > 0) & (e < 1)).all():
        raise ValueError("Assignment probabilities must be strictly between zero and one")
    scores = m.astype(float).copy()
    scores[:, 0] += (t == 0) * (y - m[:, 0]) / (1 - e)
    scores[:, 1] += (t == 1) * (y - m[:, 1]) / e
    return scores  # DR scores can fall outside [0, 1]; do not clip them.


def rule_scores(counterfactuals, choices):
    return counterfactuals[:, :1] + choices * (counterfactuals[:, 1:] - counterfactuals[:, :1])


def decision_rules(predictions):
    n = len(next(iter(predictions["Application"].values())))
    named = {"Always four examples": np.zeros(n, dtype=np.int8), "Always one example": np.ones(n, dtype=np.int8)}
    for label, stage in [("Application", "Application"), ("Two-stage", "Two-stage borrowing"), ("Direct", "Direct borrowing")]:
        for (features, method), p in predictions[stage].items():
            named[f"{label} | {features} | {method}"] = (100 * (p[:, 1]-p[:, 0]) > 1e-8).astype(np.int8)
    unique, positions, aliases = [], {}, []
    for label, choices in named.items():
        key = choices.tobytes()
        if key not in positions:
            positions[key] = len(unique)
            unique.append(choices)
        i = positions[key]
        aliases.append({"Rule label": label, "Unique rule": f"R{i+1:02d}", "Unique position": i})
    return np.column_stack(unique), pd.DataFrame(aliases)


def evaluate_policies(train, frame, predictions, repetitions=2000):
    choices, aliases = decision_rules(predictions)
    lookup = aliases.set_index("Rule label")["Unique position"].to_dict()
    e, counts, strata = assignment_probabilities(train, frame)
    e_risk, _, _ = assignment_probabilities(train, frame, ("risk",))
    y, t = (frame[name].to_numpy(dtype=int) for name in [LOAN, TREATMENT])
    m = predictions["Direct borrowing"]["history", "XGB S"]
    m_log = predictions["Direct borrowing"]["baseline", "Logistic S"]
    scores = rule_scores(counterfactual_scores(y, t, e, m), choices)
    values = scores.mean(axis=0)
    sensitivities = [rule_scores(counterfactual_scores(y, t, ee, mm), choices).mean(axis=0) for ee, mm in
                     [(e, np.zeros_like(m)), (e_risk, m), (e, m_log)]]
    children = np.random.SeedSequence(20261106).spawn(2)
    record_rng, assignment_rng = map(np.random.default_rng, children)
    boots = np.empty((repetitions, choices.shape[1]))
    partial = np.empty_like(boots)
    ns, es = counts.training_records.to_numpy(dtype=int), counts.estimated_p_one.to_numpy()
    for b in range(repetitions):
        ix = record_rng.integers(0, len(frame), size=len(frame))
        boots[b] = scores[ix].mean(axis=0)
        ones = assignment_rng.binomial(ns, es)
        if not ((ones > 0) & (ones < ns)).all():
            raise ValueError("A resampled group lost one of the two letter formats.")
        partial[b] = rule_scores(counterfactual_scores(y[ix], t[ix], (ones/ns)[strata[ix]], m[ix]), choices[ix]).mean(axis=0)
        if (b + 1) % 500 == 0:
            print(f"Completed {b+1}/{repetitions} paired policy resamples.", flush=True)

    def pair(a, b):
        return [1000*(values[a]-values[b]), *np.quantile(1000*(boots[:, a]-boots[:, b]), [.025, .975]),
                *np.quantile(1000*(partial[:, a]-partial[:, b]), [.025, .975])]
    four, one = lookup["Always four examples"], lookup["Always one example"]
    value_rows, sensitivity_rows = [], []
    for i in range(choices.shape[1]):
        labels = aliases.loc[aliases["Unique position"].eq(i), "Rule label"].tolist()
        vs_four, vs_one = pair(i, four), pair(i, one)
        value_rows.append({"Rule": f"R{i+1:02d}", "Representative label": labels[0], "Equivalent labels": len(labels),
                           "Chooses one (%)": 100*choices[:, i].mean(), "DR loans / 1,000": 1000*values[i],
                           "Gain vs all-four / 1,000": vs_four[0], "Conditional 95% low vs four": vs_four[1], "Conditional 95% high vs four": vs_four[2],
                           "Gain vs all-one / 1,000": vs_one[0], "Conditional 95% low vs one": vs_one[1], "Conditional 95% high vs one": vs_one[2]})
        sensitivity_rows.append({"Rule": f"R{i+1:02d}", "Representative label": labels[0], "Primary DR gain vs four / 1,000": vs_four[0],
                                 **{label: 1000*(v[i]-v[four]) for label, v in zip([
                                     "IPW risk/wave gain vs four / 1,000", "DR risk-only gain vs four / 1,000", "DR logistic nuisance gain vs four / 1,000"], sensitivities)}})
    specs = [("Always one example", "Always four examples")]
    for features in ("baseline", "history"):
        specs.extend((f"Two-stage | {features} | {method}", f"Application | {features} | {method}") for method in ("Logistic S", "XGB S", "XGB T"))
        specs.extend((f"Direct | {features} | {method}", f"Two-stage | {features} | {method}") for method in ("Logistic S", "XGB S"))
    matched = []
    for first, second in specs:
        a, b = lookup[first], lookup[second]
        point, low, high, plow, phigh = pair(a, b)
        primary = first == "Two-stage | history | Logistic S" and second == "Application | history | Logistic S"
        matched.append({"First rule": first, "Reference rule": second, "Identical choices": a == b,
                        "Difference / 1,000": point, "Conditional 95% low": low, "Conditional 95% high": high,
                        "Partial-resampling 2.5%": plow, "Partial-resampling 97.5%": phigh, "Role": "PRIMARY" if primary else "Secondary"})
    return {"primary_result": [r for r in matched if r["Role"] == "PRIMARY"],
            "matched_comparisons": matched, "policy_values": value_rows,
            "estimator_sensitivities": sensitivity_rows, "rule_aliases": aliases.to_dict("records")}
