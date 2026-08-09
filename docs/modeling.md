# Forecasting Methodology

## Forecast design

The target is daily SKU-store demand for the next 28 days. Boosting models use a stacked direct
design with one row per origin and horizon. This retains a single global model while preventing
the use of demand that would not be observed at the origin.

Demand features include the origin-day observation, longer seasonal lags, target-aligned seasonal
lags, trailing distribution statistics, non-zero rates, time since first and last sale, and
short-to-long trend. Price features compare planned price with origin price and its trailing
28-day mean. Future features include horizon, calendar, events, SNAP, and planned price. Every
target-aligned lag is at least as long as the 28-day horizon, so it is observed at prediction time.

## Candidates

Seasonal naive and a 28-day moving average establish transparent controls. LightGBM and XGBoost
use Tweedie regression. LightGBM uses native categorical splits after stable ordinal encoding and
stays on its portable CPU wheel. XGBoost and N-HiTS run on CPU
in the development image and use CUDA in the optional `full` image. N-HiTS uses 168 days of
context, robust scaling, known future variables, and quantile loss for `q05`, `q50`, and `q95`.

The boosting search uses a deterministic ten percent sample. Hyperparameters are selected on a
temporal validation origin and the candidate is refitted on the complete profile. N-HiTS uses a
fixed reviewed architecture so the execution budget remains comparable.

Tree forecasts apply one bounded demand-level factor learned on the temporal validation origin.
The factor is the ratio of observed to predicted validation demand, clipped to `[0.8, 1.25]`.
It is stored with the fitted model and applied before residual interval calibration. This corrects
systematic level bias without inspecting the evaluation origin.

## Backtesting

The three folds advance by 28 days and never mix future observations into training. WRMSSE uses
the revenue weights and scale denominators available at each origin. Metrics are persisted by
fold, hierarchy level, horizon, and model.

WRMSSE remains the primary M5 metric because it evaluates all hierarchy levels and weights them by
recent revenue. WAPE reports total absolute error relative to total demand and remains defined for
zero-demand rows. MAE and RMSE express point error in units, with RMSE assigning more weight to
large misses. Signed bias exposes systematic under- or over-forecasting. Coverage measures the
share of observations inside `q05` to `q95`, while mean interval width indicates how much
uncertainty was required to obtain that coverage. Maximum fold degradation captures temporal
instability relative to seasonal naive.

WAPE is also reported at SKU-store-day, 28-day SKU-store, store-day, store-category-day,
total-day, weekly SKU-store, and weekly store-category levels. Aggregation happens before metric
calculation within each fold. The default operational dashboard level is store-day, while the
bottom-level result always remains visible.

## Improvement study

The focused improvement runner in `scripts/benchmark_tree_objectives.py` holds out `d_1913` and
uses only the preceding temporal origin for post-processing choices. It compares count objectives,
absolute-error objectives, native categories, a hurdle model, seasonal ensembles, and hierarchical
calibration without installing additional model libraries. L1 and median objectives lowered WAPE
by forecasting too many zeros and failed the five percent bias guardrail. Native categories gave a
small valid improvement; hurdle, blends, and hierarchical calibration did not produce a material
bottom-level gain. Temporal demand-level calibration was retained because it removed systematic
bias and improved hierarchy-weighted accuracy.

## Explainability

TreeExplainer produces SHAP global and local attributions for both tree candidates. N-HiTS uses
Integrated Gradients through NeuralForecast. Attribution values from different explanation
methods are not compared numerically.

Review focuses on stability across horizon, state, category, and demand intermittency. Unexpected
dependence on identifiers, missing-price flags, or a single event family is treated as a model
risk even when aggregate accuracy improves.
