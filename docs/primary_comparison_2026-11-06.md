# Final reserved-test comparison

*Written on November 6, 2026, before the test set was scored, and kept as written. File and variable names refer to my original working notebook.*

Recorded November 6, 2026, after development probability, decision-value, calibration, and selected-model sensitivity results were seen, before any reserved-test model or policy scoring. This is a development-informed final specification, not a preregistration made before seeing the data. Full-file aggregate outcomes were already inspected before partitioning; this limitation remains explicit.

## Question and primary contrast

Does using both application and conditional successful-borrowing probabilities improve the letter decisions made from application probabilities alone?

The primary final contrast is **history-feature logistic two-stage minus history-feature logistic application-based decisions**, in successful loans per 1,000 eligible test records. History was the designated primary feature extension before the bounded searches, and logistic includes explicit format-by-feature interactions in both stages. These already fitted constructions can give different format recommendations. This choice addresses the central funnel question; it was not selected for the largest positive development gain (the history development estimate was -0.211 per 1,000). Their development stability results and the rationale for selecting this single contrast must be disclosed.

The logistic contrast is not presumed to be the best model or a causal effect on approval. It is a prespecified comparison of two frozen decision rules. The direct history logistic rule versus the two-stage history logistic rule is a secondary architecture comparison. All other already specified rule/reference comparisons are descriptive secondary checks rather than additional primary tests.

## Frozen objects and decision rule

Keep the original 31,916-record training fits and their selected settings. Do not refit on training plus validation, recalibrate using validation, retune, alter feature definitions, average bootstrap models, or select a favorable seed for this assessment. Use the unchanged public-data checksum, original grouped partition and seeds, baseline/history feature definitions, and original selected-model dictionaries. The 30 bootstrap fits were diagnostics, not a new deployed ensemble.

For each original fit, score both formats on the reserved 10,639-record test partition. Choose one example only when its predicted contrast exceeds 1e-8 percentage points; otherwise choose four. Retain both constant rules, six application rules, six two-stage rules, and four direct rules, and identify identical assignment vectors on this cohort. Equality of validation vectors need not imply equality of test vectors. Do not choose a winning rule by the largest test value.

Use separate names such as `final_test_predictions` and `final_policy_values`. Never replace `validation_df` with `test_df` to reuse a validation cell; preserve all development objects and their provenance.

## Probability checks

Report actual-format application log loss/Brier and the fitting-training-average benchmark on all test records; conditional successful-borrowing scores and its training-applicant benchmark on test applicants only; overall successful-borrowing scores and its training-overall benchmark on all test records. Keep the denominators explicit. Report observed and mean predicted rates and the same fixed risk/wave/actual-format diagnostics, with counts. These checks do not by themselves validate treatment-effect predictions.

## Adjusted decision-value checks

Use the same common evaluators and working assumptions recorded in `policy_evaluation_plan.md`. Estimate risk/wave assignment probabilities from ORIGINAL training records, map them to test features, and stop for unsupported strata or zero/one probabilities. Do not estimate them using test outcomes or relabel empirical probabilities as recovered original randomization constants. The incomplete original assignment documentation remains an identification limitation even if the test results are positive.

The common primary outcome nuisance is the original training-fitted direct history XGBoost S model. Compute DR scores on all eligible test records and paired rule differences; do not evaluate a rule by its own average predicted effect. The same evaluator is used for all rules. Retain fixed risk/wave IPW, risk-only DR, and baseline-logistic-outcome DR point-estimate sensitivities. Compare each rule with both all-one and all-four; label the single primary contrast above distinctly.

Use 2,000 paired ordinary test-record bootstrap repetitions and pointwise percentile intervals conditional on all fitted functions, with `SeedSequence(20261106).spawn(2)` for test-position and assignment-count streams. Keep the separate partial-resampling sensitivity that perturbs training one-example counts within risk/wave strata while holding policies/outcome models fixed. As in development, stop on an unsupported perturbed arm; do not clip, redraw, or drop cases. Partial bounds omit full model-selection uncertainty and covariance with other training fits. Unknown repeated-client dependence and the record-level resampling assumption remain limitations.

The primary contrast has one nominal conditional 95% interval. Secondary intervals are pointwise, not simultaneous family-wise evidence; do not promote a favorable secondary result to the primary conclusion. The prior aggregate data inspection, incomplete design recovery, and development-informed specification prevent describing this as an untouched preregistered causal trial.

## Reporting and stopping

Preserve the first complete test output, package versions, selected settings, and execution metadata. If the primary interval includes zero, report that the final assessment has not established an improvement; do not infer equivalence or no effect. If it excludes zero, qualify the direction and magnitude by the stated assumptions and uncertainty limits. Algebraic zero differences for identical rules remain distinct from estimated zero effects for customers.

Do not reopen the candidate grid or alter the main specification in response to test results. New calibration models, deeper searches, custom value-based tuning, ensembles, and future campaigns belong to a separately labeled follow-up study. Final manuscript work still includes condensing the working draft and completing the reference-reading/variable-provenance review; these are not reasons to keep adding models to the current comparison.

**Status:** Specification recorded. No final-test model/policy evaluation executed by creating this document.
