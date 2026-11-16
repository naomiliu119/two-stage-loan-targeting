# Two-stage loan targeting

Code for my paper *From Applications to Completed Loans: Does Modeling the Second Stage Improve Targeting Decisions?*

Campaign models usually target customers based on who is likely to respond. But a response only matters if it turns into a completed transaction, so a natural idea is to model both steps: who applies, and which applicants actually take out a loan. I wanted to know whether adding that second step leads to better targeting decisions, not just better predictions.

The data come from a randomized field experiment by Bertrand et al. (2010), in which a South African lender mailed loan offers to former customers. Some letters showed one example loan and others showed four. The question here is which letter to send each customer.

## The model

For each customer and each letter, the model estimates:

1. **Stage 1:** the chance the customer applies (fitted on all training letters), and
2. **Stage 2:** the chance an applicant goes on to borrow (fitted on applicants only).

Multiplying the two gives the chance of a loan under each letter, and the model recommends whichever letter comes out higher. Each stage is fitted with regularized logistic regression and with XGBoost, using S- and T-learner setups for Stage 1.

To judge the recommendations, I use a doubly robust estimator, because letters weren't assigned evenly across the lender's risk groups and a raw comparison is misleading.

## What I found

- Both stages predict behavior better than chance (test AUC around 0.65 for applying and 0.69 for borrowing), and there's no sign of overfitting.
- Every version of the model does better than sending everyone the four-example letter, but none does better than sending everyone the one-example letter.
- The model can't tell which customers the letter helps most. Uplift doesn't fall off across deciles of predicted effect, and the Qini coefficients aren't distinguishable from zero.
- Adding Stage 2 doesn't produce more loans. In the comparison I fixed before looking at the test set, it changed the result by −2.39 loans per 1,000 letters (95% CI −6.83 to 1.91).

My reading is that the second stage has the most influence exactly where it has the least data: only 361 applicants in the training set didn't borrow. It's useful for understanding where the letter's effect comes from, but not for deciding who gets which letter.

## Running the code

You'll need Python 3.12 or 3.13.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python run_analysis.py
```

`run_analysis.py` downloads the public data, checks it against a known checksum, recreates the train/validation/test split, fits the models, and evaluates them on the test set. At the end it compares the results with my original run (saved in `results/reference/`) and stops if anything doesn't match. Outputs go to a new folder under `results/runs/`.

The two scripts below produce the extra metrics and figures in the paper:

```bash
python -m scripts.describe_models     # coefficients, SHAP attribution, overfitting check
python -m scripts.targeting_metrics   # AUC/KS by stage, uplift deciles, Qini
python -m scripts.plot_deciles        # decile figure
python -m scripts.plot_diagrams       # funnel and model diagrams
```

Tests:

```bash
python -m unittest discover -s tests
```

On a Mac, XGBoost may need the OpenMP runtime (`brew install libomp`).

## Layout

```
run_analysis.py        main reproduction
src/                   data loading, models, evaluation, tuning search, plots
scripts/               extra diagnostics and figures for the paper
tests/                 small hand-calculated checks
results/reference/     results from my original run, used for the checks
docs/                  the primary comparison, as written before the test set was scored
```

## Data

The data are public and aren't included here; the code downloads them from the [ExperimentData mirror](https://github.com/gsbDBI/ExperimentData/tree/master/Advertising) and checks the file's SHA-256 hash. I use the 53,194 letters from the second and third mailing rounds, since the letter format wasn't recorded in the first. The original replication files are on [Harvard Dataverse](https://doi.org/10.7910/DVN/II4HDS).

A few things to keep in mind: the original assignment probabilities weren't published, so I estimate them within risk group and mailing round; the outcome is whether a customer took out a loan, not whether it was profitable or repaid; and there's no customer ID, so repeat customers can't be identified.

## References

- Bertrand, M., Karlan, D., Mullainathan, S., Shafir, E., & Zinman, J. (2010). What's advertising content worth? Evidence from a consumer credit marketing field experiment. *Quarterly Journal of Economics*, 125(1), 263–306.
- Dudík, M., Langford, J., & Li, L. (2011). Doubly robust policy evaluation and learning. *ICML*.
- Künzel, S. R., Sekhon, J. S., Bickel, P. J., & Yu, B. (2019). Metalearners for estimating heterogeneous treatment effects using machine learning. *PNAS*, 116(10), 4156–4165.

## License

MIT. See [LICENSE](LICENSE).
