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

The reproduced development champion has 72.21% SKU-store-day WAPE and 15.15% store-day WAPE.
The latter supports aggregate store planning, but aggregation does not make the item-level result
production-ready. More than half of evaluated SKU-days have zero demand, while dense-series WAPE
remains 57.9%; the error is therefore not attributable to intermittency alone.

An August 2026 improvement study tested Tweedie variance powers, Poisson, Huber, L1, median
quantile regression, native LightGBM categories, a hurdle occurrence/amount model, five-week
seasonal means, blends, and temporal calibration by series and hierarchy. On the untouched
`d_1913` fold, L1 reached 64.81% WAPE only by producing -33.82% bias. The best bias-aware
specialized candidates remained near 69%. None met a 20% bottom-level target. A separate
leakage-safe level calibration brought XGBoost bias from -5.88% to -0.82%; the resulting model
reached WRMSSE 0.7827, passed all promotion guardrails, and became Registry version 5.

The champion's mean pinball loss is 0.308 demand units and its unweighted bottom-level RMSSE is
0.733. Its inventory policy reaches 96.90% mean fill rate, 0.93% stockout rate, 25.84 average
units on hand, 2,030.7 lost-sale units, and 14,912.75 total simulated cost. These figures are
reported together because improving cost can trade away service.

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
