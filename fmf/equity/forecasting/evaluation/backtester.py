"""ExpandingWindowBacktester orchestrator (S10).

Per-row PIT-correct expanding-window backtester. See
plans/2026-06-07-s10-backtester.md for the load-bearing decisions
(target definition, fold cutoff semantics, purge mechanism, OOS
meta-learner training, feature cap).
"""

from __future__ import annotations

import dataclasses as _dc
import datetime as dt
import logging
import uuid
from dataclasses import dataclass, field

import duckdb
import numpy as np
import pandas as pd

from fmf.equity.forecasting.evaluation._backtester_config import BacktesterConfig
from fmf.equity.forecasting.evaluation._feature_cap import top_k_feature_importance
from fmf.equity.forecasting.evaluation._fold_generator import FoldSpec, generate_folds
from fmf.equity.forecasting.evaluation._target_lookup import (
    TargetRecord,
    last_fy_actual,
    next_fy_target,
)
from fmf.equity.forecasting.evaluation.prediction_cache import (
    PredictionCache,
    data_fingerprint,
    derive_cache_key,
)
from fmf.equity.forecasting.models.baselines import (
    fit_ar1_pooled,
    last_year_actual_baseline,
    predict_ar1,
    random_walk,
    seasonal_naive,
)
from fmf.equity.forecasting.models.lightgbm_model import LightGBMForecaster
from fmf.equity.forecasting.models.meta_learner import MetaLearner
from fmf.equity.forecasting.models.tirex_model import TirexBackend, TirexForecaster
from fmf.features.as_of_grid import (
    AsOfSample,
    filing_dates_grid,
    fiscal_year_end_grid,
    quarterly_grid,
)
from fmf.features.builtin_features import BUILTIN_REGISTRY
from fmf.features.feature_registry import FeatureRegistry, compute_feature_matrix
from fmf.features.point_in_time import fetch_consensus_pit, fetch_prices_pit

log = logging.getLogger(__name__)

GRID_FUNCTIONS = {
    "filing_dates": filing_dates_grid,
    "fiscal_year_end": fiscal_year_end_grid,
    "quarterly": quarterly_grid,
}

_CACHE_MODEL_NAMES = ("LightGBM", "TiRex", "NaiveLastYear", "Ensemble")


@dataclass(frozen=True, slots=True)
class BacktestRow:
    fold_idx: int
    cutoff: dt.date
    cutoff_label: str
    security_id: uuid.UUID
    symbol: str
    as_of_date: dt.date
    target_fy: int
    target_accepted_date: dt.date
    horizon_days: int
    target_value: float
    naive_baseline: float | None
    lgbm_pred: float | None
    tirex_pred: float | None
    ensemble_pred: float | None
    ensemble_source: str
    yf_consensus_snapshot: float | None
    rw_pred: float | None = None
    sn_pred: float | None = None
    ly_pred: float | None = None
    ar1_pred: float | None = None


@dataclass
class FoldDiagnostics:
    fold_idx: int
    cutoff: dt.date
    is_seed: bool
    train_n: int
    test_n: int
    train_target_accepted_max: dt.date | None
    train_keys: frozenset[tuple[uuid.UUID, dt.date]]
    test_keys: frozenset[tuple[uuid.UUID, dt.date]]
    selected_features: frozenset[str] | None = None
    meta_active: bool = False
    meta_train_source_folds: frozenset[int] | None = None
    ar1_phi: float | None = None
    ar1_train_pairs_n: int | None = None
    cache_hits: int | None = None
    cache_total: int | None = None


@dataclass
class BacktestResult:
    config: BacktesterConfig
    folds: list[FoldSpec]
    rows: list[BacktestRow] = field(default_factory=list)
    fold_diagnostics: dict[int, FoldDiagnostics] = field(default_factory=dict)
    unresolved_target_count: int = 0
    naive_baseline_missing_count: int = 0
    meta_learned_from_fold: int | None = None

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([_dc.asdict(r) for r in self.rows])


class ExpandingWindowBacktester:
    """Per-row PIT-correct expanding-window backtester.

    Decisions 1-11 in plans/2026-06-07-s10-backtester.md are load-bearing
    for correctness; do not change them without re-running the close-read
    gate.
    """

    def __init__(
        self,
        conn: duckdb.DuckDBPyConnection,
        config: BacktesterConfig,
        *,
        tirex_backend: TirexBackend | None = None,
        cache: PredictionCache | None = None,
    ) -> None:
        self._conn = conn
        self._config = config
        self._tirex = TirexForecaster(backend=tirex_backend) if tirex_backend is not None else None
        self._cache = cache
        self._fingerprint_cache: dict[uuid.UUID, str] = {}
        self._config_flags_hash: str | None = None

    def run(self, security_ids: list[uuid.UUID]) -> BacktestResult:
        folds = generate_folds(self._config.start_year, self._config.end_year)
        result = BacktestResult(config=self._config, folds=folds)
        candidate_rows = self._materialize_candidates(security_ids)
        candidate_rows = self._attach_targets_and_baseline(candidate_rows, result)
        for fold in folds:
            self._run_fold(fold, candidate_rows, result)
        return result

    def _run_fold(self, fold: FoldSpec, candidates: pd.DataFrame, result: BacktestResult) -> None:
        cutoff = fold.cutoff
        window_end = fold.window_end
        train_mask = candidates["target_accepted_date"] < cutoff
        test_mask = (candidates["as_of_date"] >= cutoff) & (candidates["as_of_date"] < window_end)
        train_rows = candidates[train_mask].copy()
        test_rows = candidates[test_mask].copy()
        diag = FoldDiagnostics(
            fold_idx=fold.fold_idx,
            cutoff=cutoff,
            is_seed=fold.is_seed,
            train_n=len(train_rows),
            test_n=len(test_rows),
            train_target_accepted_max=(
                train_rows["target_accepted_date"].max() if len(train_rows) else None
            ),
            train_keys=frozenset(
                zip(train_rows["security_id"], train_rows["as_of_date"], strict=True)
            ),
            test_keys=frozenset(
                zip(test_rows["security_id"], test_rows["as_of_date"], strict=True)
            ),
        )
        result.fold_diagnostics[fold.fold_idx] = diag
        if fold.is_seed or len(train_rows) < self._config.min_train_samples or len(test_rows) == 0:
            return
        # Cache probe: all-or-nothing per fold. If every (test_row, model_name)
        # entry is already cached, skip the compute path entirely and emit
        # BacktestRow records from cached values. Any miss falls through to the
        # full compute path, which writes all entries back at fold end.
        per_row_keys: list[dict[str, str]] | None = None
        if self._cache is not None:
            per_row_keys = self._build_cache_keys(test_rows)
            all_keys = [k for keys in per_row_keys for k in keys.values()]
            diag.cache_total = len(all_keys)
            hits = sum(1 for k in all_keys if self._cache.contains(k))
            diag.cache_hits = hits
            if hits == diag.cache_total and diag.cache_total > 0:
                for i, (_, r) in enumerate(test_rows.iterrows()):
                    result.rows.append(
                        self._make_backtest_row_from_cache(fold=fold, row=r, keys=per_row_keys[i])
                    )
                return
        # Build features (per-fold, train rows only -> per-fold ranking).
        train_X = self._build_features(train_rows)
        test_X = self._build_features(test_rows)
        train_y = train_rows["target_value"].to_numpy(dtype=np.float64)
        # Drop train rows with any-NaN target. (Features may have NaN; LightGBM
        # handles those natively.)
        finite_y = np.isfinite(train_y)
        if int(finite_y.sum()) < self._config.min_train_samples:
            return
        train_X = train_X.loc[finite_y].reset_index(drop=True)
        train_y_finite = train_y[finite_y]
        # Per-fold pooled AR(1) on (y, y_lag) pairs from this fold's training
        # rows only. Mirrors the feature-cap PIT discipline.
        ar1_pairs = train_rows.loc[finite_y, ["target_value", "naive_baseline"]].rename(
            columns={"target_value": "y", "naive_baseline": "y_lag"}
        )
        phi = fit_ar1_pooled(ar1_pairs)
        diag.ar1_phi = phi
        diag.ar1_train_pairs_n = int(
            ar1_pairs.dropna()
            .pipe(lambda d: d[np.isfinite(d["y"]) & np.isfinite(d["y_lag"])])
            .shape[0]
        )
        # LightGBM (Decision 10 ranking happens immediately after fit, on
        # this-fold-only train_X).
        lgbm = LightGBMForecaster(seed=self._config.seed).fit(train_X, train_y_finite)
        selected_features: frozenset[str] | None = None
        if self._config.feature_cap_top_k and self._config.feature_cap_top_k < train_X.shape[1]:
            top_cols = top_k_feature_importance(lgbm, k=self._config.feature_cap_top_k)
            lgbm = LightGBMForecaster(seed=self._config.seed).fit(train_X[top_cols], train_y_finite)
            test_X = test_X[top_cols]
            selected_features = frozenset(top_cols)
        diag.selected_features = selected_features
        lgbm_test = lgbm.predict(test_X)
        # TiRex per row (or NaN if no backend injected).
        tirex_test = np.array(
            [self._tirex_for_row(r) for _, r in test_rows.iterrows()], dtype=np.float64
        )
        # Meta-learner: OOS prior-fold triples realized at this cutoff.
        oos_train = self._gather_realized_oos(result, cutoff=fold.cutoff)
        meta = None
        meta_active = False
        if len(oos_train) >= self._config.meta_min_train:
            meta = MetaLearner(consensus_floor=0.0).train(
                oos_train["lgbm_pred"].to_numpy(dtype=np.float64),
                oos_train["tirex_pred"].to_numpy(dtype=np.float64),
                oos_train["naive_baseline"].to_numpy(dtype=np.float64),
                oos_train["target_value"].to_numpy(dtype=np.float64),
            )
            meta_active = True
            if result.meta_learned_from_fold is None:
                result.meta_learned_from_fold = fold.fold_idx
        diag.meta_active = meta_active
        diag.meta_train_source_folds = (
            frozenset(int(x) for x in oos_train["source_fold_idx"].unique())
            if len(oos_train)
            else frozenset()
        )
        # Test predictions.
        per_row_preds: list[tuple[float | None, float | None, float | None, float | None]] = []
        for i, (_, r) in enumerate(test_rows.iterrows()):
            lp = float(lgbm_test[i])
            tp = float(tirex_test[i])
            n_raw = r["naive_baseline"]
            n = float(n_raw) if n_raw is not None and pd.notna(n_raw) else float("nan")
            if meta_active and meta is not None and np.isfinite(n) and np.isfinite(tp):
                ens_arr = meta.predict(np.array([lp]), np.array([tp]), np.array([n]))
                ens: float | None = float(ens_arr[0])
                source = "oos_learned"
            else:
                finite_signals = [s for s in (lp, tp, n) if np.isfinite(s)]
                ens = sum(finite_signals) / len(finite_signals) if finite_signals else None
                source = "cold_start_equal_weight"
            yf_consensus = self._fetch_yf_snapshot(r)
            prev = None if not np.isfinite(n) else n
            rw_pred = random_walk(prev)
            sn_pred = seasonal_naive(prev, season_length=1)
            ly_pred = last_year_actual_baseline(prev)
            ar1_pred = predict_ar1(phi, prev)
            lp_out = lp if np.isfinite(lp) else None
            tp_out = tp if np.isfinite(tp) else None
            n_out = None if not np.isfinite(n) else n
            per_row_preds.append((lp_out, tp_out, n_out, ens))
            result.rows.append(
                BacktestRow(
                    fold_idx=fold.fold_idx,
                    cutoff=fold.cutoff,
                    cutoff_label=fold.cutoff_label,
                    security_id=r["security_id"],
                    symbol=r["symbol"],
                    as_of_date=r["as_of_date"],
                    target_fy=int(r["target_fy"]),
                    target_accepted_date=r["target_accepted_date"],
                    horizon_days=int(r["horizon_days"]),
                    target_value=float(r["target_value"]),
                    naive_baseline=n_out,
                    lgbm_pred=lp_out,
                    tirex_pred=tp_out,
                    ensemble_pred=ens,
                    ensemble_source=source,
                    yf_consensus_snapshot=yf_consensus,
                    rw_pred=rw_pred,
                    sn_pred=sn_pred,
                    ly_pred=ly_pred,
                    ar1_pred=ar1_pred,
                )
            )
        # Cache write-through: at fold end, batch all four-model entries per
        # test row. Only fires when cache is wired AND we ran the compute path.
        if self._cache is not None and per_row_keys is not None:
            entries: list[tuple[str, float | None]] = []
            for i, preds in enumerate(per_row_preds):
                keys = per_row_keys[i]
                lp_v, tp_v, n_v, ens_v = preds
                entries.append((keys["LightGBM"], lp_v))
                entries.append((keys["TiRex"], tp_v))
                entries.append((keys["NaiveLastYear"], n_v))
                entries.append((keys["Ensemble"], ens_v))
            self._cache.put_batch(entries)

    def _build_cache_keys(self, test_rows: pd.DataFrame) -> list[dict[str, str]]:
        """Per test row, derive a key for each of the four model_names. Caches
        the config_flags_hash and per-security data_fingerprint on the instance.
        """
        from fmf.research.fmf_runs import config_flags_hash

        if self._config_flags_hash is None:
            cfg_dict = {
                k: list(v) if isinstance(v, tuple) else v
                for k, v in {
                    k: getattr(self._config, k) for k in self._config.__dataclass_fields__
                }.items()
            }
            self._config_flags_hash = config_flags_hash(cfg_dict)
        fingerprints: dict[uuid.UUID, str] = {}
        for sid in test_rows["security_id"].unique():
            if sid not in self._fingerprint_cache:
                self._fingerprint_cache[sid] = data_fingerprint(self._conn, sid)
            fingerprints[sid] = self._fingerprint_cache[sid]
        per_row_keys: list[dict[str, str]] = []
        for _, r in test_rows.iterrows():
            row_keys: dict[str, str] = {}
            for mn in _CACHE_MODEL_NAMES:
                row_keys[mn] = derive_cache_key(
                    config_flags_hash=self._config_flags_hash,
                    fingerprint=fingerprints[r["security_id"]],
                    security_id=r["security_id"],
                    as_of_date=r["as_of_date"],
                    metric=self._config.metric,
                    model_name=mn,
                )
            per_row_keys.append(row_keys)
        return per_row_keys

    def _make_backtest_row_from_cache(
        self, *, fold: FoldSpec, row: pd.Series, keys: dict[str, str]
    ) -> BacktestRow:
        """Build a BacktestRow on a full-cache-hit fold. AR(1) phi was not
        fit on this path so ar1_pred is None; rw/sn/ly collapse to the cached
        naive_baseline per S11 Decision 1.
        """
        assert self._cache is not None
        lp = self._cache.get(keys["LightGBM"])
        tp = self._cache.get(keys["TiRex"])
        n = self._cache.get(keys["NaiveLastYear"])
        ens = self._cache.get(keys["Ensemble"])
        return BacktestRow(
            fold_idx=fold.fold_idx,
            cutoff=fold.cutoff,
            cutoff_label=fold.cutoff_label,
            security_id=row["security_id"],
            symbol=row["symbol"],
            as_of_date=row["as_of_date"],
            target_fy=int(row["target_fy"]),
            target_accepted_date=row["target_accepted_date"],
            horizon_days=int(row["horizon_days"]),
            target_value=float(row["target_value"]),
            naive_baseline=n,
            lgbm_pred=lp,
            tirex_pred=tp,
            ensemble_pred=ens,
            ensemble_source="cache_hit",
            yf_consensus_snapshot=self._fetch_yf_snapshot(row),
            rw_pred=n,
            sn_pred=n,
            ly_pred=n,
            ar1_pred=None,
        )

    def _gather_realized_oos(self, result: BacktestResult, *, cutoff: dt.date) -> pd.DataFrame:
        if not result.rows:
            return pd.DataFrame(
                columns=[
                    "source_fold_idx",
                    "lgbm_pred",
                    "tirex_pred",
                    "naive_baseline",
                    "target_value",
                ]
            )
        df = pd.DataFrame(
            [
                {
                    "source_fold_idx": r.fold_idx,
                    "target_accepted_date": r.target_accepted_date,
                    "lgbm_pred": r.lgbm_pred,
                    "tirex_pred": r.tirex_pred,
                    "naive_baseline": r.naive_baseline,
                    "target_value": r.target_value,
                }
                for r in result.rows
            ]
        )
        realized = df["target_accepted_date"] < cutoff
        finite = (
            df["lgbm_pred"].notna()
            & df["tirex_pred"].notna()
            & df["naive_baseline"].notna()
            & df["target_value"].notna()
        )
        return df[realized & finite].drop(columns=["target_accepted_date"])

    def _build_features(self, rows: pd.DataFrame) -> pd.DataFrame:
        reg = self._sub_registry()
        matrix_rows: list[dict[str, float | None]] = []
        for _, r in rows.iterrows():
            values = compute_feature_matrix(
                conn=self._conn,
                reg=reg,
                security_id=r["security_id"],
                as_of_date=r["as_of_date"],
            )
            matrix_rows.append(values)
        return pd.DataFrame(matrix_rows, columns=list(self._config.feature_ids)).astype("float64")

    def _sub_registry(self) -> FeatureRegistry:
        builtin_by_name = {f.name: f for f in BUILTIN_REGISTRY}
        missing = set(self._config.feature_ids) - set(builtin_by_name.keys())
        if missing:
            raise ValueError(f"unknown feature_ids: {sorted(missing)}")
        sub = FeatureRegistry()
        for fid in self._config.feature_ids:
            sub.register(builtin_by_name[fid])
        return sub

    def _tirex_for_row(self, row: pd.Series) -> float:
        if self._tirex is None:
            return float("nan")
        prices = fetch_prices_pit(
            conn=self._conn,
            security_id=row["security_id"],
            as_of_date=row["as_of_date"],
        )
        if prices.empty or "close" not in prices.columns:
            return float("nan")
        series = prices["close"].to_numpy(dtype=np.float64)
        if int(np.count_nonzero(~np.isnan(series))) < 12:
            return float("nan")
        try:
            out = self._tirex.predict(series, horizon=4)
        except Exception:
            return float("nan")
        # QUANTILE_LEVELS = (0.1..0.9); index 4 is q=0.5 (median).
        return float(out.quantiles[-1, 4])

    def _fetch_yf_snapshot(self, row: pd.Series) -> float | None:
        df = fetch_consensus_pit(
            conn=self._conn,
            security_id=row["security_id"],
            as_of_date=row["as_of_date"],
        )
        if df.empty or "metric" not in df.columns:
            return None
        sub = df[df["metric"] == self._config.metric]
        if sub.empty or "consensus" not in sub.columns:
            return None
        return float(sub.iloc[-1]["consensus"])

    def _materialize_candidates(self, security_ids: list[uuid.UUID]) -> pd.DataFrame:
        grid_fn = GRID_FUNCTIONS[self._config.grid_strategy]
        symbols = self._fetch_symbols(security_ids)
        samples: list[AsOfSample] = []
        for sid in security_ids:
            samples.extend(grid_fn(self._conn, sid, symbols[sid]))
        return pd.DataFrame(
            [
                {
                    "security_id": s.security_id,
                    "symbol": s.symbol,
                    "as_of_date": s.as_of_date,
                    "as_of_source": s.as_of_source,
                }
                for s in samples
            ]
        )

    def _fetch_symbols(self, security_ids: list[uuid.UUID]) -> dict[uuid.UUID, str]:
        placeholders = ",".join(["?"] * len(security_ids))
        rows = self._conn.execute(
            f'SELECT security_id, symbol FROM "securities" WHERE security_id IN ({placeholders})',
            [str(s) for s in security_ids],
        ).fetchall()
        return {uuid.UUID(str(r[0])): r[1] for r in rows}

    def _attach_targets_and_baseline(
        self, rows: pd.DataFrame, result: BacktestResult
    ) -> pd.DataFrame:
        target_records: list[TargetRecord | None] = []
        baselines: list[float | None] = []
        for _, r in rows.iterrows():
            tgt = next_fy_target(
                conn=self._conn,
                security_id=r["security_id"],
                as_of_date=r["as_of_date"],
                metric=self._config.metric,
            )
            target_records.append(tgt)
            if tgt is None:
                baselines.append(None)
                continue
            baselines.append(
                last_fy_actual(
                    conn=self._conn,
                    security_id=r["security_id"],
                    as_of_date=r["as_of_date"],
                    metric=self._config.metric,
                )
            )
        rows = rows.copy()
        rows["_target"] = target_records
        rows["naive_baseline"] = baselines
        resolved_mask = rows["_target"].notna()
        result.unresolved_target_count = int((~resolved_mask).sum())
        if result.unresolved_target_count > 0 and self._config.log_unresolved_targets:
            log.warning(
                "Backtester dropped %d rows with no resolvable next-FY target",
                result.unresolved_target_count,
            )
        rows = rows[resolved_mask].copy()
        rows["target_fy"] = rows["_target"].apply(lambda t: t.fiscal_year)
        rows["target_accepted_date"] = rows["_target"].apply(lambda t: t.accepted_date)
        rows["target_value"] = rows["_target"].apply(lambda t: t.value)
        rows["horizon_days"] = (
            pd.to_datetime(rows["target_accepted_date"]) - pd.to_datetime(rows["as_of_date"])
        ).dt.days.astype(int)
        rows = rows.drop(columns=["_target"])
        result.naive_baseline_missing_count = int(rows["naive_baseline"].isna().sum())
        return rows


_SCOREBOARD_MODELS = (
    "LightGBM",
    "TiRex",
    "Ensemble",
    "RandomWalk",
    "AR1",
    "SeasonalNaive",
    "NaiveLastYear",
)
_PRED_COL = {
    "LightGBM": "lgbm_pred",
    "TiRex": "tirex_pred",
    "Ensemble": "ensemble_pred",
    "RandomWalk": "rw_pred",
    "AR1": "ar1_pred",
    "SeasonalNaive": "sn_pred",
    "NaiveLastYear": "ly_pred",
}

_HORIZON_BUCKETS = (
    ("short", 0, 200),
    ("medium", 200, 365),
    ("long", 365, 100_000),
)

_EMPTY_METRICS: dict[str, float | None] = {
    "mape": None,
    "median_ape": None,
    "accuracy_within_10pct": None,
    "accuracy_within_25pct": None,
    "directional_accuracy": None,
    "beat_miss_accuracy": None,
    "coverage": 0.0,
    "correlation": None,
}


def _score_one(sub: pd.DataFrame, pred_col: str, total_attempts: int) -> dict[str, float | None]:
    from fmf.equity.forecasting.evaluation.metrics import (
        accuracy_within_pct,
        beat_miss_accuracy,
        correlation,
        coverage,
        directional_accuracy,
        mape,
        median_ape,
    )

    if sub.empty:
        return dict(_EMPTY_METRICS)
    a = sub["target_value"].to_numpy(dtype=np.float64)
    p = sub[pred_col].to_numpy(dtype=np.float64)
    prev_for_dir = sub["naive_baseline"].to_numpy(dtype=np.float64)
    return {
        "mape": mape(a, p),
        "median_ape": median_ape(a, p),
        "accuracy_within_10pct": accuracy_within_pct(a, p, threshold_pct=0.10),
        "accuracy_within_25pct": accuracy_within_pct(a, p, threshold_pct=0.25),
        "directional_accuracy": directional_accuracy(prev_for_dir, a, p),
        "beat_miss_accuracy": beat_miss_accuracy(prev_for_dir, a, p),
        "coverage": coverage(total_attempts, len(sub)),
        "correlation": correlation(a, p),
    }


def scoreboard_from_result(
    result: BacktestResult,
    *,
    by_horizon_bucket: bool = False,
) -> pd.DataFrame:
    """Per-model 7-metric scoreboard from a backtest result.

    For directional_accuracy and beat_miss_accuracy, the "prev" / "consensus"
    series is the naive_baseline (last-year actual). The v1.0 backtest has
    no historical analyst consensus by spec (Decision 7), so naive doubles
    as the comparison anchor across all model rows. Documented in the S22
    methodology doc.

    When by_horizon_bucket=True, returns a multi-index (model, bucket) frame
    with the same 7-metric columns sliced by short (<=200d), medium
    (200-365d), long (>365d). The S10 horizon distribution is long-weighted
    (median 360, p99 732); aggregate alone hides the short-horizon tail.
    See L-EVAL-S11-001.
    """
    df = result.to_frame()
    total_attempts = len(df)
    if not by_horizon_bucket:
        rows: dict[str, dict[str, float | None]] = {}
        for model in _SCOREBOARD_MODELS:
            pred_col = _PRED_COL[model]
            sub = df.dropna(subset=[pred_col, "target_value"])
            rows[model] = _score_one(sub, pred_col, total_attempts)
        return pd.DataFrame.from_dict(rows, orient="index")
    sliced: dict[tuple[str, str], dict[str, float | None]] = {}
    for model in _SCOREBOARD_MODELS:
        pred_col = _PRED_COL[model]
        model_sub = df.dropna(subset=[pred_col, "target_value"])
        for bucket, low, high in _HORIZON_BUCKETS:
            bucket_sub = model_sub[
                (model_sub["horizon_days"] >= low) & (model_sub["horizon_days"] < high)
            ]
            bucket_total = int(((df["horizon_days"] >= low) & (df["horizon_days"] < high)).sum())
            sliced[(model, bucket)] = _score_one(bucket_sub, pred_col, bucket_total)
    index = pd.MultiIndex.from_tuples(list(sliced.keys()), names=["model", "bucket"])
    return pd.DataFrame(list(sliced.values()), index=index)
