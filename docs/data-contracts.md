# Data Contracts

## Source

The required M5 files are `calendar.csv`, `sales_train_evaluation.csv`,
`sales_train_validation.csv`, `sample_submission.csv`, and `sell_prices.csv`. The preflight checks
required columns before Spark starts. Bronze records `_source_file`, `_source_sha256`,
`_ingested_at`, and `_run_id` on every table.

## Silver sales_daily

The natural key is `(series_id, day_num)`. `series_id` is the concatenation of item and store.
The table contains the original retail hierarchy, `date`, calendar events, SNAP flags, weekly
price, demand units, and `price_missing`. Demand must be non-negative and calendar coverage must
be complete.

## Gold training_features

The natural key is `(series_id, origin_day, horizon)`. `target` is observed demand at
`origin_day + horizon`. Every demand feature is calculated strictly before `origin_day`.
Calendar, event, SNAP, and price fields refer to the target day because they are assumed known at
forecast time.

## Gold forecasts_bottom

Each row contains run and model identifiers, origin and target dates, the complete retail
hierarchy, horizon, point forecast, and `q05`, `q50`, `q95`. Forecasts and quantiles are
non-negative and quantiles must be monotonic.

## Gold inventory outputs

`inventory_daily` contains demand, sales, lost sales, arrivals, stock position, orders, and costs.
`inventory_kpis` summarizes service and cost by series and policy model.
`inventory_recommendations` contains reorder point, order-up-to level, current stock assumption,
and suggested order quantity. `assumption_source` identifies synthetic inputs.
