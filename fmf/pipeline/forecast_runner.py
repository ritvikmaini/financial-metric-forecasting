"""Pipeline-stage 2: run inference + write predictions parquet."""

from __future__ import annotations

import hashlib
import uuid
from pathlib import Path

import pandas as pd

from fmf.equity.forecasting.models.lightgbm_model import LightGBMForecaster

_NAMESPACE = uuid.UUID("5f2e8b73-2c91-4a17-9e58-a7d3c1f4b602")


def run_forecast(
    *,
    model_path: str | Path,
    dataset: pd.DataFrame,
    metric: str,
    output_dir: str | Path = "reports/predictions",
    model_name: str = "LightGBM",
) -> tuple[Path, uuid.UUID]:
    """Load model, predict on dataset, write parquet, return (path, run_id).

    run_id is deterministic via UUID5 over (model_path, as_of_date,
    security_ids_hash, metric).
    """
    model = LightGBMForecaster.load_model(model_path)
    feature_cols = model.feature_names()
    missing = set(feature_cols) - set(dataset.columns)
    if missing:
        raise ValueError(
            f"dataset missing required features: {sorted(missing)}; got {sorted(dataset.columns)}"
        )
    X = dataset[feature_cols]
    preds = model.predict(X)
    as_of_date = dataset["as_of_date"].iloc[0]
    security_ids = sorted(dataset["security_id"].astype(str).tolist())
    ids_hash = hashlib.sha256("|".join(security_ids).encode()).hexdigest()[:16]
    run_id = uuid.uuid5(_NAMESPACE, f"{model_path}|{as_of_date}|{ids_hash}|{metric}")
    out = pd.DataFrame(
        {
            "security_id": dataset["security_id"].astype(str),
            "symbol": dataset["symbol"],
            "as_of_date": dataset["as_of_date"],
            "metric": metric,
            "prediction": preds,
            "model_name": model_name,
            "run_id": str(run_id),
        }
    )
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.parquet"
    out.to_parquet(out_path, index=False)
    return out_path, run_id
