"""Research-entry CLI: run the S10 backtester and record to the run registry."""

from __future__ import annotations

import datetime as dt
import logging
import uuid
from pathlib import Path

import duckdb
import typer

from fmf.equity.forecasting.evaluation._backtester_config import BacktesterConfig
from fmf.equity.forecasting.evaluation.backtester import ExpandingWindowBacktester
from fmf.research.fmf_runs import Registry, RunRecord, config_flags_hash, run_id_for

REPO_ROOT = Path(__file__).parent.parent
DEFAULT_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "mini.duckdb"
DEFAULT_REGISTRY = REPO_ROOT / "reports" / "fmf_runs.db"

app = typer.Typer(add_completion=False)


def _utc_now() -> dt.datetime:
    """Monkeypatchable wall-clock for idempotency tests that span minute boundaries."""
    return dt.datetime.now(dt.UTC)


@app.command("run")
def cli_run(
    metric: str = typer.Option("eps_diluted", "--metric"),
    start_year: int = typer.Option(2020, "--start-year"),
    end_year: int = typer.Option(2022, "--end-year"),
    feature: list[str] = typer.Option(..., "--feature"),
    fixture: Path = typer.Option(DEFAULT_FIXTURE, "--fixture"),
    registry: Path = typer.Option(DEFAULT_REGISTRY, "--registry"),
    grid_strategy: str = typer.Option("filing_dates", "--grid-strategy"),
    min_train_samples: int = typer.Option(10, "--min-train-samples"),
) -> None:
    cfg = BacktesterConfig(
        metric=metric,  # type: ignore[arg-type]
        start_year=start_year,
        end_year=end_year,
        grid_strategy=grid_strategy,  # type: ignore[arg-type]
        feature_ids=tuple(feature),
        min_train_samples=min_train_samples,
    )
    conn = duckdb.connect(str(fixture), read_only=True)
    started_at = _utc_now()
    try:
        ids_rows = conn.execute('SELECT security_id FROM "securities" ORDER BY symbol').fetchall()
        ids = [uuid.UUID(str(r[0])) for r in ids_rows]
        # TiRex backend is intentionally None in this CLI - the smoke
        # entry-point is for fast registry exercise, not production scoring.
        bt = ExpandingWindowBacktester(conn, cfg, tirex_backend=None)
        result = bt.run(ids)
    finally:
        conn.close()
    finished_at = _utc_now()
    config = {k: getattr(cfg, k) for k in cfg.__dataclass_fields__}
    cfg_hash = config_flags_hash({k: v for k, v in config.items()})
    started_minute = started_at.replace(second=0, microsecond=0).isoformat()
    rid = run_id_for(
        mode="adhoc",
        config_flags_hash_value=cfg_hash,
        window=f"{start_year}-{end_year}",
        metric=metric,
        started_at_iso_minute=started_minute,
    )
    record = RunRecord(
        run_id=rid,
        mode="adhoc",
        config_flags_hash=cfg_hash,
        commit_sha=None,
        metric=metric,
        start_year=start_year,
        end_year=end_year,
        n_securities=len(ids),
        n_rows_scored=len(result.rows),
        status="ok",
        started_at=started_at,
        finished_at=finished_at,
        config={k: list(v) if isinstance(v, tuple) else v for k, v in config.items()},
    )
    reg = Registry(registry)
    try:
        inserted = reg.record_run(record)
    finally:
        reg.close()
    typer.echo(f"run_id={rid} inserted={inserted} rows={len(result.rows)}")


@app.command("list-recent")
def cli_list_recent(
    registry: Path = typer.Option(DEFAULT_REGISTRY, "--registry"),
    metric: str | None = typer.Option(None, "--metric"),
    limit: int = typer.Option(10, "--limit"),
) -> None:
    reg = Registry(registry)
    try:
        runs = reg.list_runs(mode="adhoc", metric=metric)
    finally:
        reg.close()
    if not runs:
        typer.echo("no runs found")
        return
    for r in runs[:limit]:
        typer.echo(
            f"run_id={r.run_id} metric={r.metric} "
            f"years={r.start_year}-{r.end_year} "
            f"rows={r.n_rows_scored or 0} started={r.started_at.isoformat()}"
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    app()
