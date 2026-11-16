"""Reproduce the paper's main results from the public data.

Downloads and checks the data, fits the selected models on the training set,
evaluates them on the test set, and compares the results with the original
run saved in results/reference/.
"""
import argparse
from datetime import datetime, timezone
import json
from hashlib import sha256
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import sklearn
import xgboost

from src.data import SOURCE_SHA256, index_digest, load_data, split_data
from src.diagnostics import prediction_intervals, refit_stability
from src.evaluation import calibration_bins, decision_rules, evaluate_policies, probability_scores
from src.models import fit_selected, predict_all
from src.plots import calibration_figure, policy_figure

ROOT = Path(__file__).resolve().parent


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def verify_reference(report):
    """Compare this run with the original saved results."""
    reference = json.loads((ROOT / "results/reference/original_final_aggregates.json").read_text())
    checks = []
    for table, keys, cols, tol in [
        ("probability_scores", ["Approach", "Feature set", "Model"], ["Log loss", "Brier", "Mean predicted (%)"], 2e-6),
        ("matched_comparisons", ["First rule", "Reference rule"], ["Difference / 1,000", "Conditional 95% low", "Conditional 95% high", "Partial-resampling 2.5%", "Partial-resampling 97.5%"], .02),
        ("policy_values", ["Rule"], ["DR loans / 1,000", "Chooses one (%)"], .02),
    ]:
        actual = pd.DataFrame(report[table]).set_index(keys).sort_index()
        expected = pd.DataFrame(reference[table]).set_index(keys).sort_index()
        if not actual.index.equals(expected.index):
            raise AssertionError(f"{table} rows differ from the original results.")
        error = float(np.max(np.abs(actual[cols].to_numpy()-expected[cols].to_numpy())))
        if error > tol:
            raise AssertionError(f"{table}: max difference {error} is above {tol}; check package versions.")
        checks.append({"table": table, "maximum_absolute_difference": error, "tolerance": tol})
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/raw/loan_marketing_experiment.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "results/runs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument("--search", action="store_true", help="Also rerun the 104-setting tuning search (slow).")
    parser.add_argument("--refits", action="store_true", help="Also rerun the 30 bootstrap refits for the stability check (slow).")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Output folder already exists: {args.output}")
    args.output.mkdir(parents=True)
    raw = load_data(args.data)
    partitions = split_data(raw)
    print("Data checked; partition sizes:", {k: len(v) for k, v in partitions.items()}, flush=True)
    train, validation, test = (partitions[k] for k in ["training", "validation", "test"])
    if args.search:
        from src.search import reproduce_search
        search = reproduce_search(train)
        (args.output / "training_cv.json").write_text(json.dumps(search, indent=2, default=json_default, allow_nan=False) + "\n")
    models = fit_selected(train)
    dev_predictions, test_predictions = predict_all(models, validation), predict_all(models, test)
    # Compare the recommendation for every test letter, not just the totals.
    vectors, aliases = decision_rules(test_predictions)
    choice_hashes = {row["Rule label"]: sha256(vectors[:, row["Unique position"]].astype(np.int8).tobytes()).hexdigest()
                     for row in aliases.to_dict("records")}
    expected_hashes = json.loads((ROOT / "results/reference/original_choice_hashes.json").read_text())
    if choice_hashes != expected_hashes:
        raise AssertionError("Recommendations differ from the original run; check package versions.")
    scores = probability_scores(train, test, test_predictions)
    dev_scores = probability_scores(train, validation, dev_predictions)
    bins = calibration_bins(validation, dev_predictions)
    intervals = prediction_intervals(train, test, test_predictions)
    interval_keys = ["Kind", "Approach", "Feature set", "Model", "Metric"]
    original_intervals = pd.read_json(ROOT / "results/reference/original_prediction_intervals.json").set_index(interval_keys).sort_index()
    reproduced_intervals = intervals.set_index(interval_keys).sort_index()
    if not original_intervals.index.equals(reproduced_intervals.index):
        raise AssertionError("The set of prediction-score comparisons changed.")
    interval_columns = ["Difference", "Conditional 95% low", "Conditional 95% high"]
    interval_error = float(np.max(np.abs(original_intervals[interval_columns].to_numpy()-reproduced_intervals[interval_columns].to_numpy())))
    if interval_error > 2e-6:
        raise AssertionError(f"Prediction-score intervals differ from the original by {interval_error}")
    policies = evaluate_policies(train, test, test_predictions)
    report = {"metadata": {"status": "Reproduction of the recorded analysis, not a new independent test",
                          "raw_data_sha256": SOURCE_SHA256,
                          "partitions": {k: {"records": len(v), "index_sha256": index_digest(v)} for k, v in partitions.items()},
                          "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__, "xgboost": xgboost.__version__},
                          "uncertainty": "2,000 paired test-record resamples; fitted functions fixed. Separate partial bounds perturb assignment counts only.",
                          "assignment": "Training-estimated risk/wave shares; original probabilities not recovered."},
              "probability_scores": scores.to_dict("records"), **policies}
    checks = verify_reference(report)
    checks.extend([{"table": "18 full recommendation vectors", "all_sha256_match": True},
                   {"table": "48 prediction-score comparisons", "maximum_absolute_difference": interval_error, "tolerance": 2e-6}])
    report["reproduction_checks"] = checks
    for name, table in [("final_probability_scores", scores), ("development_probability_scores", dev_scores), ("development_calibration_bins", bins),
                        ("paired_prediction_intervals", intervals),
                        *[(name, pd.DataFrame(policies[name])) for name in ["primary_result", "policy_values", "matched_comparisons", "estimator_sensitivities", "rule_aliases"]]]:
        table.to_csv(args.output / f"{name}.csv", index=False)
    (args.output / "reproduction.json").write_text(json.dumps(report, indent=2, default=json_default, allow_nan=False) + "\n")
    calibration_figure(bins, args.output / "development_calibration_logistic.png")
    calibration_figure(bins, args.output / "development_calibration_xgb.png", "XGB S")
    policy_figure(pd.DataFrame(policies["matched_comparisons"]), args.output / "two_stage_vs_application.png")
    if args.refits:
        refit_stability(train, validation, dev_predictions).to_csv(args.output / "training_refit_stability.csv", index=False)
    print(pd.DataFrame(policies["primary_result"]).to_string(index=False), flush=True)
    print("All checks passed. Outputs saved to:", args.output, flush=True)


if __name__ == "__main__":
    main()
