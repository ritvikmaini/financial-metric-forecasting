"""fmf-runs CLI: list, show, verify."""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import cast

import typer

from fmf.research.fmf_runs import Mode, Registry

DEFAULT_REGISTRY = Path("reports/fmf_runs.db")

app = typer.Typer(add_completion=False)

_VALID_MODES = ("adhoc", "backfill")


@app.command("list")
def cli_list(
    registry: Path = typer.Option(DEFAULT_REGISTRY, "--registry"),
    mode: str | None = typer.Option(None, "--mode"),
    metric: str | None = typer.Option(None, "--metric"),
) -> None:
    mode_typed: Mode | None = None
    if mode is not None:
        if mode not in _VALID_MODES:
            typer.echo(f"invalid mode {mode!r}; expected one of {_VALID_MODES}", err=True)
            raise typer.Exit(2)
        mode_typed = cast(Mode, mode)
    reg = Registry(registry)
    try:
        runs = reg.list_runs(mode=mode_typed, metric=metric)
    finally:
        reg.close()
    for r in runs:
        typer.echo(
            f"{r.run_id} | {r.mode} | {r.metric} | "
            f"{r.start_year}-{r.end_year} | {r.n_rows_scored} rows | {r.status}"
        )


@app.command("show")
def cli_show(
    run_id: str = typer.Argument(...),
    registry: Path = typer.Option(DEFAULT_REGISTRY, "--registry"),
) -> None:
    reg = Registry(registry)
    try:
        record = reg.get_run(uuid.UUID(run_id))
    finally:
        reg.close()
    if record is None:
        typer.echo(f"run_id {run_id} not found", err=True)
        raise typer.Exit(1)
    out = {
        "run_id": str(record.run_id),
        "mode": record.mode,
        "metric": record.metric,
        "config_flags_hash": record.config_flags_hash,
        "status": record.status,
        "started_at": record.started_at.isoformat(),
        "finished_at": (record.finished_at.isoformat() if record.finished_at else None),
        "config": record.config,
    }
    typer.echo(json.dumps(out, indent=2, sort_keys=True, default=str))


@app.command("verify")
def cli_verify(
    registry: Path = typer.Option(DEFAULT_REGISTRY, "--registry"),
) -> None:
    reg = Registry(registry)
    try:
        mismatches = reg.verify()
    finally:
        reg.close()
    if mismatches:
        for run_id, stored, recomputed in mismatches:
            typer.echo(f"{run_id}: stored={stored} recomputed={recomputed}", err=True)
        raise typer.Exit(1)
    typer.echo("registry verified: no hash drift")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    app()
