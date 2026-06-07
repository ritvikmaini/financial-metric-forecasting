"""Pipeline CLI: build dataset, run forecast, quality-check, or chain all three.

Usage:
  python scripts/run_pipeline.py build-dataset --as-of 2024-05-15 --feature revenue_ttm --feature gross_margin ...
  python scripts/run_pipeline.py forecast --model-path reports/models/lgbm_eps --dataset-path reports/datasets/...
  python scripts/run_pipeline.py quality-check --predictions-path reports/predictions/<run_id>.parquet
  python scripts/run_pipeline.py chain --as-of 2024-05-15 --model-path reports/models/lgbm_eps --feature revenue_ttm ...
"""

from __future__ import annotations

import datetime as dt
import logging
import sys
import uuid
from pathlib import Path

import duckdb
import typer

from fmf.pipeline.dataset_builder import build_inference_dataset
from fmf.pipeline.forecast_runner import run_forecast
from fmf.pipeline.quality_checks import any_failed, run_quality_checks
from fmf.research.fmf_runs import Registry, RunRecord, config_flags_hash, run_id_for

REPO_ROOT = Path(__file__).parent.parent
DEFAULT_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "mini.duckdb"
DEFAULT_REGISTRY = REPO_ROOT / "reports" / "fmf_runs.db"

app = typer.Typer(add_completion=False)
log = logging.getLogger("run_pipeline")


@app.command("build-dataset")
def cli_build_dataset(
    as_of: str = typer.Option(..., "--as-of", help="ISO date"),
    feature: list[str] = typer.Option(..., "--feature", help="repeatable feature_id"),
    fixture: Path = typer.Option(DEFAULT_FIXTURE, "--fixture"),
    output: Path = typer.Option(Path("reports/datasets/inference.parquet"), "--output"),
) -> None:
    conn = duckdb.connect(str(fixture), read_only=True)
    try:
        rows = conn.execute('SELECT security_id FROM "securities" ORDER BY symbol').fetchall()
        ids = [uuid.UUID(str(r[0])) for r in rows]
        as_of_date = dt.date.fromisoformat(as_of)
        df = build_inference_dataset(
            conn=conn,
            security_ids=ids,
            as_of_date=as_of_date,
            feature_ids=feature,
        )
    finally:
        conn.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, index=False)
    log.info("wrote %d rows to %s", len(df), output)


@app.command("forecast")
def cli_forecast(
    model_path: Path = typer.Option(..., "--model-path"),
    dataset_path: Path = typer.Option(..., "--dataset-path"),
    metric: str = typer.Option("eps_diluted", "--metric"),
) -> None:
    import pandas as pd

    dataset = pd.read_parquet(dataset_path)
    out_path, run_id = run_forecast(
        model_path=model_path,
        dataset=dataset,
        metric=metric,
    )
    log.info("wrote predictions to %s (run_id=%s)", out_path, run_id)


@app.command("quality-check")
def cli_quality_check(
    predictions_path: Path = typer.Option(..., "--predictions-path"),
    expected_security_count: int | None = typer.Option(None, "--expected-securities"),
    nan_threshold: float = typer.Option(0.05, "--nan-threshold"),
    baseline_path: Path | None = typer.Option(None, "--baseline-path"),
) -> None:
    results = run_quality_checks(
        predictions_path=predictions_path,
        expected_security_count=expected_security_count,
        nan_threshold=nan_threshold,
        baseline_path=baseline_path,
    )
    for r in results:
        prefix = "PASS" if r.passed else "FAIL"
        log.info("[%s] %s: %s", prefix, r.check_name, r.message)
    if any_failed(results):
        sys.exit(1)


@app.command("chain")
def cli_chain(
    as_of: str = typer.Option(..., "--as-of"),
    feature: list[str] = typer.Option(..., "--feature"),
    model_path: Path = typer.Option(..., "--model-path"),
    metric: str = typer.Option("eps_diluted", "--metric"),
    fixture: Path = typer.Option(DEFAULT_FIXTURE, "--fixture"),
    registry: Path = typer.Option(DEFAULT_REGISTRY, "--registry"),
    no_registry: bool = typer.Option(False, "--no-registry"),
) -> None:
    started_at = dt.datetime.now(dt.UTC)
    conn = duckdb.connect(str(fixture), read_only=True)
    try:
        rows = conn.execute('SELECT security_id FROM "securities" ORDER BY symbol').fetchall()
        ids = [uuid.UUID(str(r[0])) for r in rows]
        as_of_date = dt.date.fromisoformat(as_of)
        dataset = build_inference_dataset(
            conn=conn,
            security_ids=ids,
            as_of_date=as_of_date,
            feature_ids=feature,
        )
    finally:
        conn.close()
    out_path, run_id = run_forecast(
        model_path=model_path,
        dataset=dataset,
        metric=metric,
    )
    finished_at = dt.datetime.now(dt.UTC)
    log.info("chain wrote %s (run_id=%s)", out_path, run_id)
    if not no_registry:
        config = {
            "metric": metric,
            "as_of": as_of,
            "feature_ids": list(feature),
            "model_path": str(model_path),
            "fixture": str(fixture),
        }
        cfg_hash = config_flags_hash(config)
        started_minute = started_at.replace(second=0, microsecond=0).isoformat()
        rid = run_id_for(
            mode="adhoc",
            config_flags_hash_value=cfg_hash,
            window=as_of,
            metric=metric,
            started_at_iso_minute=started_minute,
        )
        record = RunRecord(
            run_id=rid,
            mode="adhoc",
            config_flags_hash=cfg_hash,
            commit_sha=None,
            metric=metric,
            start_year=None,
            end_year=None,
            n_securities=len(ids),
            n_rows_scored=len(dataset),
            status="ok",
            started_at=started_at,
            finished_at=finished_at,
            config=config,
        )
        reg = Registry(registry)
        try:
            inserted = reg.record_run(record)
        finally:
            reg.close()
        log.info("registry run_id=%s inserted=%s", rid, inserted)
    results = run_quality_checks(
        predictions_path=out_path,
        expected_security_count=len(ids),
    )
    for r in results:
        prefix = "PASS" if r.passed else "FAIL"
        log.info("[%s] %s: %s", prefix, r.check_name, r.message)
    if any_failed(results):
        sys.exit(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    app()
