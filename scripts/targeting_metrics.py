"""AUC and KS for each stage, and how well the model ranks customers by the
effect of the letter (risk-adjusted uplift by decile and the Qini coefficient).

Uplift uses the same doubly robust scores as the main evaluation, so it accounts
for letters being assigned unevenly across risk groups.

Run from the repository root:  python -m scripts.targeting_metrics
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from src.data import APPLICATION, LOAN, TREATMENT, load_data, split_data
from src.evaluation import assignment_probabilities, counterfactual_scores
from src.models import fit_selected, predict_all
from scripts.describe_models import verify_against_saved

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/supplementary"
LABEL = {"Logistic S": "Logistic", "XGB S": "XGBoost S", "XGB T": "XGBoost T"}
TIE_SEED = 20261116  # fixed random tie-break for ranking only


def ks(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def ranking_metrics(parts, models):
    rows = []
    for name, frame in parts.items():
        pred = predict_all(models, frame)
        t = frame[TREATMENT].to_numpy(dtype=int)
        applied = frame[APPLICATION].to_numpy(dtype=int)
        loan = frame[LOAN].to_numpy(dtype=int)
        for stage, key, y, mask in [("Stage 1: applied", "Application", applied, np.ones(len(frame), bool)),
                                    ("Stage 2: borrowed, given application", "Conditional borrowing", loan, applied == 1),
                                    ("Combined: took up a loan", "Two-stage borrowing", loan, np.ones(len(frame), bool))]:
            for (fs, method), probs in pred[key].items():
                p = probs[np.arange(len(frame)), t][mask]
                yy = y[mask]
                rows.append({"Data set": name, "Stage": stage, "Feature set": fs, "Model": LABEL[method],
                             "Records": int(mask.sum()), "Events": int(yy.sum()),
                             "AUC": roc_auc_score(yy, p), "KS": ks(yy, p)})
    return pd.DataFrame(rows)


def qini_and_deciles(effect, phi_diff, order):
    """effect: predicted uplift; phi_diff: per-letter DR uplift score (phi1-phi0)."""
    n = len(effect)
    d = phi_diff[order]
    cum = np.cumsum(d)  # extra loans from targeting the top k with the one-example letter
    k = np.arange(1, n + 1)
    random_line = k / n * cum[-1]
    qini = float(np.mean(cum - random_line)) / n * 1000  # loans per 1,000 letters
    deciles = np.array_split(np.arange(n), 10)
    dec = [float(d[ix].mean()) * 100 for ix in deciles]  # percentage points
    tops = {f"Top {q}%": float(d[: int(round(n * q / 100))].mean()) * 100 for q in (10, 20, 30)}
    return qini, dec, tops, cum


def targeting_metrics(train, test, models, repetitions=2000):
    pred = predict_all(models, test)
    y, t = test[LOAN].to_numpy(dtype=int), test[TREATMENT].to_numpy(dtype=int)
    e, _, _ = assignment_probabilities(train, test)
    m = pred["Direct borrowing"]["history", "XGB S"]
    phi = counterfactual_scores(y, t, e, m)
    phi_diff = phi[:, 1] - phi[:, 0]
    candidates = {f"Two-stage model, {LABEL[k[1]]}": pred["Two-stage borrowing"][k] for k in pred["Two-stage borrowing"] if k[0] == "history"}
    candidates["Stage 1 alone, Logistic"] = pred["Application"]["history", "Logistic S"]
    rng_tie = np.random.default_rng(TIE_SEED)
    boot_rng = np.random.default_rng(np.random.SeedSequence(20261116).spawn(1)[0])
    boot_ix = [boot_rng.integers(0, len(test), len(test)) for _ in range(repetitions)]
    summary, decile_rows, curves = [], [], {}
    for name, probs in candidates.items():
        effect = probs[:, 1] - probs[:, 0]
        tie_key = rng_tie.random(len(effect))
        order = np.lexsort((tie_key, -effect))
        qini, dec, tops, cum = qini_and_deciles(effect, phi_diff, order)
        boots = []
        for ix in boot_ix:
            o = np.lexsort((tie_key[ix], -effect[ix]))
            boots.append(qini_and_deciles(effect[ix], phi_diff[ix], o)[0])
        lo, hi = np.quantile(boots, [.025, .975])
        summary.append({"Model": name, "Distinct predicted effects": int(len(np.unique(np.round(effect, 12)))),
                        "Qini coefficient (loans per 1,000)": qini, "Qini 95% low": lo, "Qini 95% high": hi,
                        **{f"{k} adjusted uplift (pp)": v for k, v in tops.items()},
                        "All letters adjusted uplift (pp)": float(phi_diff.mean()) * 100})
        groups = np.array_split(np.arange(len(effect)), 10)
        dec_rng = np.random.default_rng(np.random.SeedSequence(20261117).spawn(1)[0])
        for i, v in enumerate(dec, 1):
            scores = phi_diff[order][groups[i - 1]]
            draws = [scores[dec_rng.integers(0, len(scores), len(scores))].mean() * 100 for _ in range(repetitions)]
            low, high = np.quantile(draws, [.025, .975])
            decile_rows.append({"Model": name, "Decile (1 = largest predicted effect)": i,
                                "Mean predicted effect (pp)": float(effect[order][groups[i - 1]].mean()) * 100,
                                "Adjusted uplift (pp)": v, "95% low (pp)": low, "95% high (pp)": high})
        curves[name] = cum
    return pd.DataFrame(summary), pd.DataFrame(decile_rows), curves


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    parts = split_data(load_data())
    train, test = parts["training"], parts["test"]
    models = fit_selected(train)
    diff, _ = verify_against_saved(train, test, models)
    print(f"Refitted models match the original test scores (max difference {diff:.1e}).")
    ranking_metrics(parts, models).to_csv(OUT / "ranking_metrics_auc_ks.csv", index=False)
    summary, deciles, curves = targeting_metrics(train, test, models)
    summary.to_csv(OUT / "qini_summary_test.csv", index=False)
    deciles.to_csv(OUT / "uplift_deciles_test.csv", index=False)
    print("Wrote ranking_metrics_auc_ks.csv, qini_summary_test.csv, uplift_deciles_test.csv")


if __name__ == "__main__":
    main()
