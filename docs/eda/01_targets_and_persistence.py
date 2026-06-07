# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # EDA 01: Targets and persistence
#
# This is the keystone notebook of the EDA series. The forecasting story
# stands or falls on whether the targets we are predicting -- annual EPS,
# EBIT, and EBITDA -- carry enough year-over-year persistence that a
# naive predictor is a hard baseline to beat. If the target is highly
# persistent, naive-last-year is a strong baseline, which is exactly why
# a tree model does not trivially beat it on a thin cross-section and
# exactly why a per-cell noise floor (S15, deferred) is needed to
# separate a real small win from sample noise.
#
# Three questions, three answers, one conclusion that ties to
# `docs/FORECASTING.md` §5a.
#
# **Universe.** 9 anchor tickers from the committed fixture
# (`tests/fixtures/mini.duckdb`): AAPL, GOOGL, GWW, HSY, JNJ, JPM, MSFT,
# SNOW, ZTS. Annual disclosures span FY2007 to FY2026 across the set;
# per-ticker coverage varies (SNOW from 2020, ZTS from 2011, others from
# 2007 to 2010).
#
# **Honesty about cross-section thinness.** Nine tickers is small. The
# pooled lag-1 correlation on 100+ (y_t, y_{t-1}) pairs is robust; the
# per-ticker lag-1 ACF on 7-19 points per ticker is noisy; the
# pre/post-COVID 2019 vs 2021 comparison is borderline-underpowered with
# N=8 per side. Each finding is qualified in line with its sample size.

# %%
from __future__ import annotations

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


# Resolve repo root by walking up from cwd until we see the marker.
def _repo_root() -> Path:
    p = Path.cwd().resolve()
    for parent in (p, *p.parents):
        if (parent / "pyproject.toml").exists() and (parent / "fmf").exists():
            return parent
    raise RuntimeError(f"could not find fmf-public repo root from cwd={p}")


REPO_ROOT = _repo_root()
FIG_DIR = REPO_ROOT / "docs" / "eda" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "mini.duckdb"

plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.dpi"] = 110
plt.rcParams["font.size"] = 10
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

NEAR_ZERO_QUANTILE = 0.05
NEAR_ZERO_ABS_FLOOR = 1e-3


def load_annual(metric: str) -> pd.DataFrame:
    """First-disclosure FY value per (security, fiscal_year) for `metric`.

    Mirrors the comparative-row-safe target-lookup logic from S10: pick
    MIN(accepted_date) per (security, fiscal_year) on non-null rows, then
    join back to get the value at that earliest disclosure.
    """
    conn = duckdb.connect(str(FIXTURE), read_only=True)
    try:
        df = conn.execute(
            f"""
            WITH fy_first AS (
                SELECT security_id, fiscal_year, MIN(accepted_date) AS first_accepted
                FROM income_statement
                WHERE period = 'FY' AND "{metric}" IS NOT NULL
                GROUP BY security_id, fiscal_year
            )
            SELECT s.symbol AS symbol,
                   fy.fiscal_year AS fiscal_year,
                   fy.first_accepted AS first_accepted,
                   ist."{metric}" AS value,
                   ist.end_date AS end_date
            FROM fy_first fy
            JOIN income_statement ist
              ON ist.security_id = fy.security_id
              AND ist.fiscal_year = fy.fiscal_year
              AND ist.period = 'FY'
              AND ist.accepted_date = fy.first_accepted
              AND ist."{metric}" IS NOT NULL
            JOIN securities s ON s.security_id = fy.security_id
            ORDER BY symbol, fiscal_year
            """
        ).fetchdf()
    finally:
        conn.close()
    return df


eps_df = load_annual("eps_diluted")
ebit_df = load_annual("ebit")
ebitda_df = load_annual("ebitda")

coverage = (
    pd.DataFrame(
        {
            "eps_diluted": eps_df.groupby("symbol").size(),
            "ebit": ebit_df.groupby("symbol").size(),
            "ebitda": ebitda_df.groupby("symbol").size(),
        }
    )
    .fillna(0)
    .astype(int)
)
coverage.loc["TOTAL"] = coverage.sum()
coverage

# %% [markdown]
# **Data-shape finding to surface up front.** EBITDA carries zero
# (security, fiscal_year) disclosures across the fixture. EBITDA is a
# non-GAAP measure and rarely receives an XBRL tag; EDGAR-tagged
# fundamentals are GAAP-anchored. The proprietary system used a
# CompanyFacts-derived EBITDA proxy (`net_income + interest + tax +
# depreciation_and_amortization`); v1.0 does not. The persistence
# analysis below proceeds on `eps_diluted` (125 pairs across 8 tickers;
# HSY excluded by the `eps_diluted` source tag gap from
# `IDEA-S22-001`) and `ebit` (134 pairs across 9 tickers including HSY).
# Wherever the text says "the target", read it as eps_diluted unless
# explicitly comparing the two.

# %% [markdown]
# ## Question 1: How persistent are annual EPS and EBIT year over year?
#
# Two angles, because they measure different things.
#
# 1. **Lag-1 autocorrelation in levels** measures whether `y_t` correlates
#    with `y_{t-1}` directly. High correlation means a naive predictor
#    has a small variance to fight. Trend in levels inflates this number
#    mechanically (companies grow), so the next angle is the honest one
#    for the naive-baseline story.
# 2. **Year-over-year percent change distribution** measures the size of
#    the move a naive predictor would have to absorb each year. The
#    median absolute percent change here is *exactly* the lower bound
#    on naive-last-year MedAPE on this universe -- if the typical YoY
#    change is 12%, no naive predictor can do better than 12% MedAPE.


# %%
def pooled_lag1_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Per (symbol, fiscal_year) row, attach y_{t-1} via the same-ticker
    prior fiscal_year. Returns the (y, y_lag, yoy_pct) frame in long form.
    """
    df = df.sort_values(["symbol", "fiscal_year"]).copy()
    df["y_lag"] = df.groupby("symbol")["value"].shift(1)
    df["yoy_pct"] = (df["value"] - df["y_lag"]) / df["y_lag"].abs()
    return df.dropna(subset=["y_lag"])


def per_ticker_lag1_acf(pairs: pd.DataFrame) -> pd.DataFrame:
    """Per-ticker lag-1 correlation in levels. Returns one row per ticker."""
    rows = []
    for sym, sub in pairs.groupby("symbol"):
        if len(sub) < 4:
            # ACF on three pairs is meaningless; skip with a note.
            rows.append({"symbol": sym, "n_pairs": len(sub), "acf_lag1": np.nan})
            continue
        rho, _ = stats.pearsonr(sub["y_lag"].to_numpy(), sub["value"].to_numpy())
        rows.append({"symbol": sym, "n_pairs": len(sub), "acf_lag1": float(rho)})
    return pd.DataFrame(rows)


eps_pairs = pooled_lag1_pairs(eps_df)
ebit_pairs = pooled_lag1_pairs(ebit_df)

eps_acf = per_ticker_lag1_acf(eps_pairs)
ebit_acf = per_ticker_lag1_acf(ebit_pairs)


def pooled_lag1_rho(pairs: pd.DataFrame) -> tuple[float, int]:
    pairs = pairs.dropna(subset=["y_lag", "value"])
    rho, _ = stats.pearsonr(pairs["y_lag"].to_numpy(), pairs["value"].to_numpy())
    return float(rho), len(pairs)


eps_pooled_rho, eps_n = pooled_lag1_rho(eps_pairs)
ebit_pooled_rho, ebit_n = pooled_lag1_rho(ebit_pairs)

print(f"eps_diluted pooled lag-1 rho = {eps_pooled_rho:.3f} (N={eps_n} pairs)")
print(f"ebit        pooled lag-1 rho = {ebit_pooled_rho:.3f} (N={ebit_n} pairs)")
print()
print("Per-ticker lag-1 ACF (in levels):")
display(
    pd.concat(
        [eps_acf.assign(metric="eps_diluted"), ebit_acf.assign(metric="ebit")], ignore_index=True
    )
)

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

for ax, (metric, pairs, pooled_rho) in zip(
    axes,
    [("eps_diluted", eps_pairs, eps_pooled_rho), ("ebit", ebit_pairs, ebit_pooled_rho)],
):
    sub = pairs.dropna(subset=["y_lag", "value"]).copy()
    # Highlight the GOOGL 20-for-1 stock split (July 2022) point so the
    # eps_diluted panel's structural break is visible, not a mystery.
    is_split = (sub["symbol"] == "GOOGL") & (sub["fiscal_year"] == 2022) & (metric == "eps_diluted")
    ax.scatter(
        sub.loc[~is_split, "y_lag"], sub.loc[~is_split, "value"], s=22, alpha=0.6, edgecolor="none"
    )
    if is_split.any():
        ax.scatter(
            sub.loc[is_split, "y_lag"],
            sub.loc[is_split, "value"],
            s=72,
            color="red",
            marker="x",
            label="GOOGL FY2022 20-for-1 split",
        )
        ax.legend(loc="lower right", frameon=False, fontsize=9)
    lo = float(min(sub["y_lag"].min(), sub["value"].min()))
    hi = float(max(sub["y_lag"].max(), sub["value"].max()))
    pad = (hi - lo) * 0.05
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="0.5", lw=1, ls="--")
    ax.set_xlabel(f"{metric} at fiscal_year - 1")
    ax.set_ylabel(f"{metric} at fiscal_year")
    ax.set_title(f"{metric}: pooled lag-1 rho = {pooled_rho:.3f}  (N={len(sub)})")

fig.suptitle("Annual target levels: y_t against y_{t-1} across the 8/9-ticker universe", y=1.02)
fig.tight_layout()
fig.savefig(FIG_DIR / "01_lag1_scatter.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# **Question 1, angle 1: lag-1 ACF in levels.** Pooled across 117 pairs,
# `eps_diluted` has rho = 0.642 -- moderate but not extreme -- while
# `ebit` has rho = 0.976 across 125 pairs. The gap matters and the
# per-ticker numbers explain it. For EBIT every ticker except JNJ
# (-0.099, n=10, likely R&D and litigation swings) and HSY (0.49) shows
# rho > 0.84, and MSFT, ZTS, GOOGL, SNOW, and AAPL all sit at 0.95+.
# EBIT is a near-pure level signal: where the business was last year is
# where the business will be next year, plus a slow drift. EPS by
# contrast varies more across tickers (0.25 GOOGL, 0.43 SNOW, 0.44 JNJ
# at the low end; 0.95+ for GWW, MSFT, ZTS at the high end). Two
# mechanisms explain the EPS spread. One, share-count actions: EPS
# divides net income by diluted shares, and the buyback programs at AAPL
# and others move the denominator independently of the business
# (AAPL rho = 0.65 reflects this). Two, the GOOGL 20-for-1 stock split
# of July 2022: GOOGL EPS goes from $112.20 (FY2021) to $4.56 (FY2022),
# a structural break that no level-correlation will tolerate (marked
# with the red X on the scatter). This is a real-data finding worth
# pulling out -- corporate actions inject discontinuities into per-share
# metrics that an academic split-adjusted backtest would smooth away.
# fmf-public uses EDGAR as-disclosed values, and the next cell confirms
# the EDA and the backtester treat the split identically: at the three
# 2022 cutoffs (Q1 May-15, Q2 Aug-14, Q3 Nov-15) the backtester's naive
# baseline for GOOGL is the pre-split FY2021 EPS of $112.20 and the
# realized target is the post-split FY2022 EPS of $4.56, producing a
# single-row naive APE of 23.6 on each of three scored rows. The headline
# MedAPE absorbs these because it is median-robust; the MAPE does not,
# which is part of why `FORECASTING.md` §4a now names the GOOGL split
# alongside the SNOW gaps as the aggregate-MAPE outlier sources. The
# as-disclosed vs split-adjusted choice itself is a stated v1.0
# limitation (`FORECASTING.md` §5i); the v1.x split-adjustment work is
# filed as `IDEA-S21-002`.

# %%
# Consistency check: what does the backtester see at the GOOGL split
# cutoffs? Reproduces the EDA finding through the backtester's actual
# target-lookup helpers, so EDA and headline trace to one source.
import datetime as dt

from fmf.equity.forecasting.evaluation._target_lookup import (
    last_fy_actual,
    next_fy_target,
)

_conn = duckdb.connect(str(FIXTURE), read_only=True)
_googl_sid = _conn.execute("SELECT security_id FROM securities WHERE symbol='GOOGL'").fetchone()[0]
_cutoffs = [dt.date(2022, 5, 15), dt.date(2022, 8, 14), dt.date(2022, 11, 15)]
_rows = []
for D in _cutoffs:
    tgt = next_fy_target(
        conn=_conn,
        security_id=_googl_sid,
        as_of_date=D,
        metric="eps_diluted",
    )
    naive = last_fy_actual(
        conn=_conn,
        security_id=_googl_sid,
        as_of_date=D,
        metric="eps_diluted",
    )
    _rows.append(
        {
            "as_of": D,
            "target_fy": tgt.fiscal_year if tgt else None,
            "target_eps": tgt.value if tgt else None,
            "naive_last_disclosed_eps": naive,
            "naive_ape": (
                abs(tgt.value - naive) / abs(tgt.value) if (tgt and naive is not None) else None
            ),
        }
    )
_conn.close()
pd.DataFrame(_rows)


# %%
def near_zero_mask(actuals: np.ndarray) -> np.ndarray:
    pos_abs = np.abs(actuals)
    pos = pos_abs[pos_abs > 0]
    if len(pos) == 0:
        return np.zeros(len(actuals), dtype=bool)
    q = float(np.quantile(pos, NEAR_ZERO_QUANTILE))
    threshold = max(q, NEAR_ZERO_ABS_FLOOR)
    return pos_abs >= threshold


def yoy_pct_stats(pairs: pd.DataFrame) -> dict[str, float]:
    pairs = pairs.dropna(subset=["yoy_pct"])
    actuals = pairs["value"].to_numpy(dtype=np.float64)
    mask = near_zero_mask(actuals)
    arr = pairs.loc[mask, "yoy_pct"].to_numpy(dtype=np.float64)
    abs_arr = np.abs(arr)
    return {
        "n": int(len(arr)),
        "median_pct": float(np.median(arr)),
        "median_abs_pct": float(np.median(abs_arr)),
        "p25_abs_pct": float(np.quantile(abs_arr, 0.25)),
        "p75_abs_pct": float(np.quantile(abs_arr, 0.75)),
        "p95_abs_pct": float(np.quantile(abs_arr, 0.95)),
    }


eps_yoy = yoy_pct_stats(eps_pairs)
ebit_yoy = yoy_pct_stats(ebit_pairs)
display(pd.DataFrame({"eps_diluted": eps_yoy, "ebit": ebit_yoy}))

# %%
fig, ax = plt.subplots(1, 1, figsize=(9, 4.5))
bins = np.linspace(-1.0, 1.5, 40)
ax.hist(
    np.clip(eps_pairs["yoy_pct"].dropna().to_numpy(), bins[0], bins[-1]),
    bins=bins,
    alpha=0.55,
    label=f"eps_diluted (N={eps_yoy['n']})",
    color="C0",
)
ax.hist(
    np.clip(ebit_pairs["yoy_pct"].dropna().to_numpy(), bins[0], bins[-1]),
    bins=bins,
    alpha=0.55,
    label=f"ebit (N={ebit_yoy['n']})",
    color="C1",
)
ax.axvline(0.0, color="0.4", lw=1)
ax.axvline(
    eps_yoy["median_pct"],
    color="C0",
    lw=1.2,
    ls="--",
    label=f"eps median = {eps_yoy['median_pct']:+.2%}",
)
ax.axvline(
    ebit_yoy["median_pct"],
    color="C1",
    lw=1.2,
    ls="--",
    label=f"ebit median = {ebit_yoy['median_pct']:+.2%}",
)
ax.set_xlabel("year-over-year percent change")
ax.set_ylabel("count")
ax.set_title("YoY percent change distribution across the fixture universe")
ax.legend(loc="upper left", frameon=False)
fig.tight_layout()
fig.savefig(FIG_DIR / "01_yoy_pct_dist.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# **Question 1, angle 2: YoY percent change distribution.** Median YoY
# change is +13.1% for EPS and +10.6% for EBIT -- companies grow on
# average, no surprise. The width is what matters for the naive
# baseline. Median absolute YoY change is 19.2% for EPS and 14.7% for
# EBIT, with p75 around 37% for EPS and 24% for EBIT and a long right
# tail (p95 EPS = 91%, EBIT = 65%). These widths are the empirical
# floor on what naive-last-year can achieve as MedAPE: roughly 19% for
# EPS, roughly 15% for EBIT. The next cell measures exactly that.

# %% [markdown]
# ## Question 2: What MedAPE does naive-last-year reach?
#
# This is the keystone number. It sets the floor a model must clear to
# claim a real win against the naive baseline.


# %%
def naive_last_year_medape(pairs: pd.DataFrame) -> dict[str, float]:
    actuals = pairs["value"].to_numpy(dtype=np.float64)
    predictions = pairs["y_lag"].to_numpy(dtype=np.float64)
    mask = near_zero_mask(actuals)
    if not mask.any():
        return {"n": 0, "medape": float("nan"), "mape": float("nan")}
    ape = np.abs((actuals[mask] - predictions[mask]) / actuals[mask])
    return {
        "n": int(mask.sum()),
        "medape": float(np.median(ape)),
        "mape": float(np.mean(ape)),
        "p10_ape": float(np.quantile(ape, 0.10)),
        "p25_ape": float(np.quantile(ape, 0.25)),
        "p75_ape": float(np.quantile(ape, 0.75)),
        "p90_ape": float(np.quantile(ape, 0.90)),
    }


eps_naive = naive_last_year_medape(eps_pairs)
ebit_naive = naive_last_year_medape(ebit_pairs)

baseline_table = pd.DataFrame({"eps_diluted": eps_naive, "ebit": ebit_naive})
display(baseline_table)

# %%
fig, ax = plt.subplots(1, 1, figsize=(7, 3.8))
ax.bar(
    ["eps_diluted", "ebit"],
    [eps_naive["medape"], ebit_naive["medape"]],
    color=["C0", "C1"],
    alpha=0.85,
    width=0.55,
)
for i, (label, stats_d) in enumerate([("eps_diluted", eps_naive), ("ebit", ebit_naive)]):
    ax.text(
        i,
        stats_d["medape"] + 0.01,
        f"{stats_d['medape']:.3f}\n(N={stats_d['n']})",
        ha="center",
        fontsize=9,
    )
ax.set_ylabel("MedAPE")
ax.set_title("Naive-last-year MedAPE per metric (the bar a model must clear)")
ax.set_ylim(0, max(eps_naive["medape"], ebit_naive["medape"]) * 1.4)
fig.tight_layout()
fig.savefig(FIG_DIR / "01_naive_medape.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# **Question 2 result, the keystone number.** Naive-last-year reaches
# MedAPE = 0.178 on EPS and 0.130 on EBIT, computed across the full
# annual history of every ticker in the fixture (not just the 2020-2023
# evaluation window). The EPS figure is the load-bearing one for the
# rest of the forecasting story: `docs/FORECASTING.md` §4 reports the
# headline backtester's Naive MedAPE at 0.180 on the 2020-2023 window,
# and this notebook recovers 0.178 on a different (wider) slice of the
# same data using the same near-zero APE filter. Two independent paths
# to the same number is the cleanest reproducibility check the fixture
# supports.
#
# That number is the floor LightGBM does not clear on the headline
# (LightGBM 0.314 vs Naive 0.180 on the 2020-2023 window). A reader
# who saw only the headline might wonder if the gap is a bug; this
# notebook's purpose is to say no, the gap is real and the explanation
# is in the data: EPS persistence is moderate (rho = 0.642) with a
# substantial cross-section spread driven by share-count effects, and
# the typical YoY change is around 19%. A small-cross-section tree
# model on four features cannot reliably out-predict naive at this
# persistence level without leakage. The §5a "honest difficulty"
# argument lands here, with the per-cell sigma (S15, deferred) as the
# tool that would classify any small win as real signal or sample
# noise.

# %% [markdown]
# ## Question 3: Pre versus post-COVID distribution shift
#
# 2019 actuals vs 2021 actuals per metric per security. With N=8 per side
# for `eps_diluted` (HSY excluded by the source tag gap) and N=9 for
# `ebit`, a KS test is borderline-underpowered. Read the figure for the
# shape; treat the p-value as suggestive.


# %%
def pre_post_covid(
    df: pd.DataFrame, metric: str
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    pre = df[df["fiscal_year"] == 2019]["value"].to_numpy(dtype=np.float64)
    post = df[df["fiscal_year"] == 2021]["value"].to_numpy(dtype=np.float64)
    if len(pre) >= 3 and len(post) >= 3:
        ks_stat, p_value = stats.ks_2samp(pre, post)
    else:
        ks_stat, p_value = float("nan"), float("nan")
    return (
        pre,
        post,
        {
            "n_pre": int(len(pre)),
            "n_post": int(len(post)),
            "ks_stat": float(ks_stat),
            "ks_p": float(p_value),
        },
    )


eps_pre, eps_post, eps_ks = pre_post_covid(eps_df, "eps_diluted")
ebit_pre, ebit_post, ebit_ks = pre_post_covid(ebit_df, "ebit")
print("KS test on 2019 vs 2021 annual actuals:")
display(pd.DataFrame({"eps_diluted": eps_ks, "ebit": ebit_ks}))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
for ax, (metric, pre, post, ks) in zip(
    axes,
    [("eps_diluted", eps_pre, eps_post, eps_ks), ("ebit", ebit_pre, ebit_post, ebit_ks)],
):
    parts = ax.violinplot([pre, post], showmedians=True, widths=0.7)
    for pc, color in zip(parts["bodies"], ["C0", "C2"]):
        pc.set_facecolor(color)
        pc.set_alpha(0.5)
    ax.set_xticks([1, 2])
    ax.set_xticklabels([f"2019 (N={ks['n_pre']})", f"2021 (N={ks['n_post']})"])
    ax.set_ylabel(metric)
    title = (
        f"{metric}: pre vs post COVID  KS p={ks['ks_p']:.3f}"
        if not np.isnan(ks["ks_p"])
        else f"{metric}: N too small for KS"
    )
    ax.set_title(title)

fig.suptitle("Pre/post-COVID distribution shift -- cross-section thinness warning", y=1.02)
fig.tight_layout()
fig.savefig(FIG_DIR / "01_pre_post_covid.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# **Question 3 result, qualified.** With N=7 to 9 per side, the KS test
# is underpowered. The measured p-values are 0.948 for EPS and 0.730 for
# EBIT -- neither comes anywhere near significance. Read this as: the
# fixture data shows no detectable level-distribution shift between 2019
# and 2021, but the sample size cannot rule out a real shift either.
# The figure is the honest deliverable: 2021 medians are visibly higher
# than 2019 medians for both metrics (recovery year for most of these
# tickers), but the within-group variance is large compared to the
# between-group shift on a sample this small. A broader EDGAR pull (a
# few hundred S&P-500 tickers across both years) would move this from
# suggestive to confirmatory; v1.0 records the limitation rather than
# overclaiming, and notebook 02 picks up the same regime-sensitivity
# question on the feature side where the per-row sample is roughly an
# order of magnitude denser.

# %% [markdown]
# ## Conclusion and tie to FORECASTING.md §5a
#
# The persistence numbers above are the empirical evidence behind the
# argument in `docs/FORECASTING.md` §5a that "LightGBM loses to
# NaiveLastYear on this fixture, and that is the honest finding." Three
# threads:
#
# 1. **EBIT is highly persistent in levels** (pooled lag-1 rho = 0.976),
#    EPS less so (rho = 0.642) because share-count programs (buybacks at
#    AAPL, GOOGL, MSFT) move the EPS denominator independently of the
#    business. Per-ticker spread is wide on EPS (0.25 to 0.97) and
#    narrow on EBIT (mostly 0.84+ with JNJ as an outlier at -0.10).
# 2. **YoY change widths set the MedAPE floor** at roughly 19% for EPS
#    and 15% for EBIT (median absolute YoY change across the fixture).
#    A naive predictor cannot do better than the width of the move it
#    has to absorb each year, and a model has to be reliably smaller
#    than that to claim a real win.
# 3. **Naive-last-year achieves MedAPE 0.178 on EPS and 0.130 on EBIT**
#    across the full annual history. The 0.178 figure here matches the
#    headline backtester's 0.180 Naive MedAPE on the 2020-2023 window
#    within reproducibility -- two paths to the same baseline. The gap
#    LightGBM did not close on the headline (0.314 vs 0.180) is
#    explained by this combination of moderate EPS persistence + 19%
#    typical YoY change width + 8 scored tickers in the cross-section.
#    A wider cross-section (`IDEA-S18-001`) and per-cell noise-floor
#    sigma (S15, deferred) are the tools that classify any small
#    LightGBM win as real signal versus sample noise.
#
# The notebook also surfaces three real-data findings that shape the
# rest of the EDA series:
#
# - **EBITDA is not directly tagged** in the fixture EDGAR data; the
#   forecasting story therefore covers EPS and EBIT only, with EBITDA
#   filed as deferred work on a CompanyFacts-derived proxy.
# - **HSY is absent from `eps_diluted`** but present in `ebit` -- a
#   selective source tag gap, not a wholesale data problem, made
#   concrete here and tracked as `IDEA-S22-001`.
# - **SNOW has very short history** (post-IPO 2020) and dominates the
#   long-horizon backtest tail. Notebook 03 will look at the filing-
#   latency distribution that drives this.
#
# Notebook 02 picks up the feature side: distributions, correlations,
# and redundancy. Notebook 03 makes the leakage story concrete by
# walking the comparative-row trap on a real (security, fiscal_year),
# the field-level coverage recovery on `total_assets`, and the HSY and
# SNOW data-source gaps end to end.
