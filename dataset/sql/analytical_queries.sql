-- ============================================================================
-- GraphGuard — Analytical Queries
-- Use these in Snowflake worksheets or integrate into feature engineering.
-- ============================================================================

-- 1. User Spending Summary
-- -------------------------------------------------------------------
-- Average, total, and count of transactions per user, with std deviation.
SELECT
    USER_ID,
    COUNT(*)                         AS total_transactions,
    ROUND(AVG(AMOUNT), 2)            AS avg_amount,
    ROUND(STDDEV(AMOUNT), 2)         AS std_amount,
    ROUND(SUM(AMOUNT), 2)            AS total_spend,
    ROUND(MIN(AMOUNT), 2)            AS min_amount,
    ROUND(MAX(AMOUNT), 2)            AS max_amount
FROM RAW_TRANSACTIONS
GROUP BY USER_ID
ORDER BY total_spend DESC;


-- 2. Spending by Category per User
-- -------------------------------------------------------------------
SELECT
    USER_ID,
    CATEGORY,
    COUNT(*)                AS txn_count,
    ROUND(SUM(AMOUNT), 2)  AS category_spend,
    ROUND(AVG(AMOUNT), 2)  AS avg_category_amount
FROM RAW_TRANSACTIONS
GROUP BY USER_ID, CATEGORY
ORDER BY USER_ID, category_spend DESC;


-- 3. Anomalous Transactions (> 2σ from user mean)
-- -------------------------------------------------------------------
-- Flags transactions whose amount exceeds user_mean + 2 * user_std.
WITH user_stats AS (
    SELECT
        USER_ID,
        AVG(AMOUNT)    AS user_avg,
        STDDEV(AMOUNT) AS user_std
    FROM RAW_TRANSACTIONS
    GROUP BY USER_ID
)
SELECT
    t.TRANSACTION_ID,
    t.USER_ID,
    t.MERCHANT_NAME,
    t.CATEGORY,
    t.AMOUNT,
    t.LOCATION,
    t.TRANSACTION_DATE,
    ROUND(s.user_avg, 2)                                AS user_avg,
    ROUND(s.user_std, 2)                                AS user_std,
    ROUND((t.AMOUNT - s.user_avg) / NULLIF(s.user_std, 0), 2) AS z_score
FROM RAW_TRANSACTIONS t
JOIN user_stats s ON t.USER_ID = s.USER_ID
WHERE (t.AMOUNT - s.user_avg) / NULLIF(s.user_std, 0) > 2
ORDER BY z_score DESC;


-- 4. Velocity Check — Users with Transaction Burst (≥ 3 in 24 h)
-- -------------------------------------------------------------------
WITH txn_windows AS (
    SELECT
        USER_ID,
        TRANSACTION_DATE,
        COUNT(*) OVER (
            PARTITION BY USER_ID
            ORDER BY TRANSACTION_DATE
            RANGE BETWEEN INTERVAL '24 HOURS' PRECEDING AND CURRENT ROW
        ) AS txn_count_24h
    FROM RAW_TRANSACTIONS
)
SELECT *
FROM txn_windows
WHERE txn_count_24h >= 3
ORDER BY USER_ID, TRANSACTION_DATE;


-- 5. Merchant Risk Profile
-- -------------------------------------------------------------------
-- Merchants with highest average transaction and most high-value txns.
SELECT
    MERCHANT_NAME,
    CATEGORY,
    COUNT(*)                AS total_txns,
    ROUND(AVG(AMOUNT), 2)  AS avg_amount,
    SUM(CASE WHEN AMOUNT > 500 THEN 1 ELSE 0 END) AS high_value_count,
    ROUND(SUM(CASE WHEN AMOUNT > 500 THEN 1 ELSE 0 END)::FLOAT / COUNT(*) * 100, 1)
        AS high_value_pct
FROM RAW_TRANSACTIONS
GROUP BY MERCHANT_NAME, CATEGORY
ORDER BY avg_amount DESC;


-- 6. Geographic Anomalies — Users Transacting in Unusual Locations
-- -------------------------------------------------------------------
WITH user_locations AS (
    SELECT
        USER_ID,
        LOCATION,
        COUNT(*) AS loc_count
    FROM RAW_TRANSACTIONS
    GROUP BY USER_ID, LOCATION
),
user_primary AS (
    SELECT
        USER_ID,
        LOCATION AS primary_location
    FROM user_locations
    QUALIFY ROW_NUMBER() OVER (PARTITION BY USER_ID ORDER BY loc_count DESC) = 1
)
SELECT
    t.TRANSACTION_ID,
    t.USER_ID,
    t.LOCATION         AS txn_location,
    p.primary_location,
    t.AMOUNT,
    t.MERCHANT_NAME,
    t.TRANSACTION_DATE
FROM RAW_TRANSACTIONS t
JOIN user_primary p ON t.USER_ID = p.USER_ID
WHERE t.LOCATION != p.primary_location
ORDER BY t.AMOUNT DESC;


-- 7. Daily Transaction Volume Trend
-- -------------------------------------------------------------------
SELECT
    DATE_TRUNC('day', TRANSACTION_DATE) AS txn_date,
    COUNT(*)                            AS daily_count,
    ROUND(SUM(AMOUNT), 2)              AS daily_volume
FROM RAW_TRANSACTIONS
GROUP BY txn_date
ORDER BY txn_date;


-- 8. Category Risk Weights (used by feature engineering)
-- -------------------------------------------------------------------
-- Higher risk for high-value categories; lower for everyday spending.
SELECT
    CATEGORY,
    CASE
        WHEN CATEGORY = 'Jewelry'      THEN 0.9
        WHEN CATEGORY = 'Electronics'  THEN 0.7
        WHEN CATEGORY = 'Clothing'     THEN 0.4
        WHEN CATEGORY = 'Retail'       THEN 0.3
        WHEN CATEGORY = 'Restaurants'  THEN 0.2
        WHEN CATEGORY = 'Grocery'      THEN 0.15
        WHEN CATEGORY = 'Gas Stations' THEN 0.1
        WHEN CATEGORY = 'Coffee Shops' THEN 0.05
        ELSE 0.3
    END AS risk_weight,
    COUNT(*)               AS txn_count,
    ROUND(AVG(AMOUNT), 2)  AS avg_amount
FROM RAW_TRANSACTIONS
GROUP BY CATEGORY
ORDER BY risk_weight DESC;
