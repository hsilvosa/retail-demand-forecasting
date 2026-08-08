SELECT
    sales.id,
    sales.series_id,
    sales.item_id,
    sales.dept_id,
    sales.cat_id,
    sales.store_id,
    sales.state_id,
    sales.d,
    sales.day_num,
    sales.units,
    calendar.date,
    calendar.wm_yr_wk,
    calendar.weekday,
    calendar.wday,
    calendar.month,
    calendar.year,
    calendar.event_name_1,
    calendar.event_type_1,
    calendar.event_name_2,
    calendar.event_type_2,
    calendar.snap_CA,
    calendar.snap_TX,
    calendar.snap_WI,
    prices.sell_price,
    prices.sell_price IS NULL AS price_missing
FROM normalized_sales AS sales
INNER JOIN silver_calendar AS calendar
    ON sales.d = calendar.d
    AND sales.day_num = calendar.day_num
LEFT JOIN silver_prices AS prices
    ON sales.store_id = prices.store_id
    AND sales.item_id = prices.item_id
    AND calendar.wm_yr_wk = prices.wm_yr_wk
