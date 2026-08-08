# Forecasting Methodology

## Forecast design

The target is daily SKU-store demand for the next 28 days. Boosting models use a stacked direct
design with one row per origin and horizon. This retains a single global model while preventing
the use of demand that would not be observed at the origin.

Demand features include the seven observations before the origin, longer seasonal lags, trailing
distribution statistics, non-zero rates, time since last sale, and short-to-long trend. Future
features include horizon, calendar, events, SNAP, and planned price.

## Candidates

Seasonal naive and a 28-day moving average establish transparent controls. LightGBM and XGBoost
use Tweedie regression and GPU acceleration. N-HiTS uses 168 days of context, robust scaling,
known future variables, and quantile loss for `q05`, `q50`, and `q95`.

The boosting search uses a deterministic ten percent sample. Hyperparameters are selected on a
temporal validation origin and the candidate is refitted on the complete profile. N-HiTS uses a
fixed reviewed architecture so the execution budget remains comparable.

## Backtesting

The three folds advance by 28 days and never mix future observations into training. WRMSSE uses
the revenue weights and scale denominators available at each origin. Metrics are persisted by
fold, hierarchy level, horizon, and model.

## Explainability

TreeExplainer produces SHAP global and local attributions for both tree candidates. N-HiTS uses
Integrated Gradients through NeuralForecast. Attribution values from different explanation
methods are not compared numerically.

Review focuses on stability across horizon, state, category, and demand intermittency. Unexpected
dependence on identifiers, missing-price flags, or a single event family is treated as a model
risk even when aggregate accuracy improves.
