-- fmf-public DuckDB schema
--
-- Mirrors the bavest.* table shape from the Bavest FMF project so the
-- point-in-time extraction logic (FMF-004 port, lands in S4) does not
-- need to change. The key invariant: every fundamentals row carries
-- accepted_date (the EDGAR `filed` field), and (security_id, fiscal_year,
-- period, accepted_date, end_date) is the primary key so amendments and
-- legitimately-distinct period-ends within the same accepted_date both
-- coexist.
--
-- This file amends the S1 schema to add end_date to the PIT-table PKs
-- (income_statement, balance_sheet, cashflow) for fiscal-calendar safety
-- — needed because the EDGAR `fy` field can label two distinct period-end
-- dates with the same (fy, fp, filed) tuple (e.g. AAPL Q1 FY2009/2010
-- straddling the fiscal-year boundary).
--
-- DuckDB does not support cross-table foreign keys natively in versions
-- this code targets; security_id integrity is enforced at the application
-- layer via the connectors module + the audit/sources scan in S2.

CREATE TABLE IF NOT EXISTS securities (
    security_id UUID PRIMARY KEY,
    symbol      TEXT NOT NULL,
    cik         TEXT NOT NULL,
    sector      TEXT,
    industry    TEXT,
    country     TEXT,
    exchange    TEXT
);

CREATE TABLE IF NOT EXISTS income_statement (
    security_id   UUID    NOT NULL,
    fiscal_year   INTEGER NOT NULL,
    period        TEXT    NOT NULL,
    filing_date   DATE    NOT NULL,
    accepted_date DATE    NOT NULL,
    end_date      DATE    NOT NULL,
    revenue                       DOUBLE,
    gross_profit                  DOUBLE,
    ebitda                        DOUBLE,
    ebit                          DOUBLE,
    net_income                    DOUBLE,
    eps_diluted                   DOUBLE,
    PRIMARY KEY (security_id, fiscal_year, period, accepted_date, end_date)
);

CREATE TABLE IF NOT EXISTS balance_sheet (
    security_id   UUID    NOT NULL,
    fiscal_year   INTEGER NOT NULL,
    period        TEXT    NOT NULL,
    filing_date   DATE    NOT NULL,
    accepted_date DATE    NOT NULL,
    end_date      DATE    NOT NULL,
    total_assets         DOUBLE,
    total_liabilities    DOUBLE,
    total_equity         DOUBLE,
    cash_and_equivalents DOUBLE,
    current_assets       DOUBLE,
    current_liabilities  DOUBLE,
    long_term_debt       DOUBLE,
    PRIMARY KEY (security_id, fiscal_year, period, accepted_date, end_date)
);

CREATE TABLE IF NOT EXISTS cashflow (
    security_id   UUID    NOT NULL,
    fiscal_year   INTEGER NOT NULL,
    period        TEXT    NOT NULL,
    filing_date   DATE    NOT NULL,
    accepted_date DATE    NOT NULL,
    end_date      DATE    NOT NULL,
    operating_cash_flow  DOUBLE,
    investing_cash_flow  DOUBLE,
    financing_cash_flow  DOUBLE,
    capital_expenditure  DOUBLE,
    free_cash_flow       DOUBLE,
    PRIMARY KEY (security_id, fiscal_year, period, accepted_date, end_date)
);

CREATE TABLE IF NOT EXISTS analyst_estimates (
    security_id UUID      NOT NULL,
    target_date DATE      NOT NULL,
    pulled_at   TIMESTAMP NOT NULL,
    metric      TEXT      NOT NULL,
    consensus   DOUBLE,
    n_analysts  INTEGER,
    PRIMARY KEY (security_id, target_date, pulled_at, metric)
);

CREATE TABLE IF NOT EXISTS prices (
    security_id UUID   NOT NULL,
    "date"      DATE   NOT NULL,
    "open"      DOUBLE,
    high        DOUBLE,
    low         DOUBLE,
    "close"     DOUBLE,
    adj_close   DOUBLE,
    volume      BIGINT,
    PRIMARY KEY (security_id, "date")
);
