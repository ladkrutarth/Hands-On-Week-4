-- ============================================================================
-- GraphGuard — Snowflake Schema Definitions
-- Run this script in your Snowflake worksheet to set up all tables.
-- ============================================================================

-- 1. Database & Schema Setup
-- -------------------------------------------------------------------
CREATE DATABASE IF NOT EXISTS GRAPHGUARD_DB;
USE DATABASE GRAPHGUARD_DB;
CREATE SCHEMA IF NOT EXISTS PUBLIC;
USE SCHEMA PUBLIC;

-- 2. Raw Transactions Table (Data Source Layer)
-- Source: Kaggle kartik2112/fraud-detection (fraudTrain.csv)
-- -------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS RAW_TRANSACTIONS (
    TRANSACTION_ID    VARCHAR(50)    PRIMARY KEY,
    USER_ID           VARCHAR(20)    NOT NULL,
    MERCHANT_NAME     VARCHAR(100),
    CARDHOLDER_NAME   VARCHAR(100),
    CATEGORY          VARCHAR(50),
    AMOUNT            FLOAT,
    LOCATION          VARCHAR(100),
    TRANSACTION_DATE  TIMESTAMP_NTZ,
    IS_FRAUD_ACTUAL   INT,           -- 0/1 ground truth from Kaggle
    STATUS            VARCHAR(20),
    INGESTED_AT       TIMESTAMP_NTZ  DEFAULT CURRENT_TIMESTAMP()
);

-- 3. Transaction Features Table (Feature Engineering Layer)
-- -------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS TRANSACTION_FEATURES (
    TRANSACTION_ID         VARCHAR(50)   PRIMARY KEY,
    USER_ID                VARCHAR(20)   NOT NULL,

    -- Amount features
    AMOUNT                 FLOAT,
    AMOUNT_ZSCORE          FLOAT,        -- deviation from user mean
    IS_HIGH_VALUE          BOOLEAN,      -- > user_mean + 2 * user_std

    -- User-level aggregates (snapshot at feature time)
    USER_AVG_AMOUNT        FLOAT,
    USER_STD_AMOUNT        FLOAT,
    USER_TOTAL_TRANSACTIONS INT,

    -- Velocity features
    TXN_COUNT_1H           INT,          -- transactions in last 1 hour
    TXN_COUNT_24H          INT,          -- transactions in last 24 hours
    TXN_COUNT_7D           INT,          -- transactions in last 7 days

    -- Category features
    CATEGORY               VARCHAR(50),
    CATEGORY_RISK_WEIGHT   FLOAT,        -- risk weight for category

    -- Time features
    HOUR_OF_DAY            INT,
    DAY_OF_WEEK            INT,
    IS_WEEKEND             BOOLEAN,

    -- Geographic features
    LOCATION               VARCHAR(100),
    IS_NEW_LOCATION        BOOLEAN,      -- not seen in user history
    LOCATION_ENTROPY       FLOAT,        -- diversity of user locations

    -- Kaggle ground truth (when available)
    IS_FRAUD_ACTUAL        INT,          -- 0 = legit, 1 = confirmed fraud
    MERCHANT_NAME          VARCHAR(100),
    CARDHOLDER_NAME        VARCHAR(100),

    COMPUTED_AT            TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

-- 4. Fraud Scores Table (Model Output Layer)
-- -------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS FRAUD_SCORES (
    SCORE_ID             VARCHAR(50)   PRIMARY KEY,
    TRANSACTION_ID       VARCHAR(50)   NOT NULL,
    USER_ID              VARCHAR(20)   NOT NULL,

    -- Rule-based scores
    ZSCORE_FLAG           FLOAT,
    VELOCITY_FLAG         FLOAT,
    CATEGORY_RISK_SCORE   FLOAT,
    GEOGRAPHIC_RISK_SCORE FLOAT,

    -- ML model scores
    ISOLATION_FOREST_SCORE FLOAT,

    -- Combined
    COMBINED_RISK_SCORE   FLOAT,        -- weighted combination, 0.0 to 1.0
    RISK_LEVEL            VARCHAR(20),  -- LOW / MEDIUM / HIGH / CRITICAL
    RECOMMENDATION        VARCHAR(200),

    -- Kaggle ground truth (when available)
    IS_FRAUD_ACTUAL        INT,          -- 0 = legit, 1 = confirmed fraud
    MERCHANT_NAME          VARCHAR(100),
    CARDHOLDER_NAME        VARCHAR(100),

    SCORED_AT             TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),

    FOREIGN KEY (TRANSACTION_ID) REFERENCES RAW_TRANSACTIONS(TRANSACTION_ID)
);

-- 5. Authentication Events Table (Decision Layer)
-- -------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS AUTH_EVENTS (
    EVENT_ID             VARCHAR(50)   PRIMARY KEY,
    USER_ID              VARCHAR(20)   NOT NULL,
    SECURITY_LEVEL       VARCHAR(20),  -- LOW / MEDIUM / HIGH / CRITICAL
    NUM_QUESTIONS        INT,
    SCORE                FLOAT,        -- authentication score 0.0–1.0
    PASSED               BOOLEAN,
    LATENCY_MS           FLOAT,
    EVENT_TIMESTAMP      TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

-- 6. Pipeline Runs Table (Monitoring Layer)
-- -------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS PIPELINE_RUNS (
    RUN_ID               VARCHAR(50)   PRIMARY KEY,
    TIMESTAMP            TIMESTAMP_NTZ NOT NULL,
    STAGE                VARCHAR(50),  -- ingestion / feature_engineering / scoring / auth
    STATUS               VARCHAR(20),  -- success / failed / success_dry_run
    RECORDS_PROCESSED    INT,
    DURATION_MS          FLOAT,
    ERROR_MESSAGE        VARCHAR(500),
    CREATED_AT           TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

-- ============================================================================
-- Indexes for performance
-- ============================================================================
-- Snowflake uses micro-partitions; explicit indexes are not needed.
-- Clustering keys can be added for large tables:

-- ALTER TABLE RAW_TRANSACTIONS CLUSTER BY (USER_ID, TRANSACTION_DATE);
-- ALTER TABLE TRANSACTION_FEATURES CLUSTER BY (USER_ID);
-- ALTER TABLE FRAUD_SCORES CLUSTER BY (RISK_LEVEL, SCORED_AT);
