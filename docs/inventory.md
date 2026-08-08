# Inventory Simulation

## Scope and assumptions

M5 does not contain observed inventory, supplier lead times, purchase orders, or cost accounting.
The simulator therefore evaluates decisions under explicit synthetic assumptions. It does not
claim to recover Walmart inventory history.

Defaults are a seven-day lead time, seven-day review period, 95 percent target service, fixed
order cost of 5, annual holding rate of 20 percent of selling price, and lost-sale penalty equal
to selling price. These values are editable in configuration and in the dashboard.

## Policy

The periodic-review `(R,S)` policy evaluates inventory position on each review day. The
order-up-to level covers lead time plus review period. One thousand residual vectors are sampled
from temporal backtests to retain dependence between forecast horizons. The requested service
quantile of simulated cumulative demand becomes the target stock position.

Orders arrive after the configured lead time. Demand above available stock is treated as lost
sales rather than backordered demand. Daily conservation checks ensure that sales plus lost sales
equals demand and that inventory remains non-negative.

## Decision metrics

Fill rate and stockout rate measure service. Average inventory, order count, holding cost,
ordering cost, lost sales, and total cost measure operating consequences. The champion policy is
always compared with seasonal naive under identical assumptions and realized demand.
