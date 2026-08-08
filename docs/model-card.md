# Model Card

## Intended use

The registered model forecasts 28 days of daily demand for M5 SKU-store series. Its intended use
is batch planning, model comparison, and inventory scenario analysis. It is not a real-time price
optimizer or an autonomous purchasing system.

## Training and evaluation

Candidates are trained on the daily M5 panel with a 730-day modeling window. Selection uses three
rolling origins and WRMSSE across the 12 official hierarchy levels. The parent MLflow run stores
the exact configuration, Git SHA, fold metrics, parameters, model artifact, and explanations.

The model version carrying the `champion` alias passed the configured WRMSSE improvement, bias,
coverage, and fold guardrails at promotion time. Exact results belong to that MLflow version and
must not be copied from a different run.

## Limitations

M5 covers ten stores in three US states and does not represent every retail format. Historical
prices and events are available, but there are no observed promotions beyond those signals,
supplier constraints, returns, inventory positions, or lost-sales labels. Zero sales can mean
zero demand or unavailable stock, and the dataset cannot distinguish them.

Future prices are treated as planned and known. Missing future price is flagged and filled for
model execution. Predictions outside the M5 calendar require a separate future-covariate source.

## Explainability

LightGBM and XGBoost use SHAP TreeExplainer. N-HiTS uses Integrated Gradients. Explanations are
diagnostic and do not establish causal effects. Identifier importance, unstable event effects,
and attribution drift across horizons require review.

## Inventory use

Inventory recommendations depend on synthetic lead time, cost, service, and on-hand assumptions.
They demonstrate decision integration and should not be interpreted as recovered Walmart policy.
Production use requires real stock, order, supplier, pack-size, and cost data plus business review.
