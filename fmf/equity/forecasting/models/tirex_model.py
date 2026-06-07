"""TiRex zero-shot time-series forecaster wrapper.

Two backends:
- TirexHuggingFaceBackend: live model from HuggingFace `NX-AI/TiRex`.
  Used by tirex-fixtures.yml CPU workflow and live.yml. NOT exercised
  in the hermetic CI lane.
- TirexFixtureBackend: replays committed JSON fixtures keyed by a hash
  of (rounded series, horizon). Used by hermetic CI tests. Raises
  FixtureNotFoundError if no fixture matches -- fail-loud, never silently
  substitute a wrong forecast.

The forecaster itself owns winsorization + monotonic enforcement; the
backend just produces raw quantiles.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from fmf.equity.forecasting.models._tirex_preprocessing import (
    ROBUST_OUTLIER_CAP_DEFAULT,
    winsorize_mad,
)
from fmf.equity.forecasting.models.protocols import TirexOutput

log = logging.getLogger(__name__)

TIREX_MODEL_NAME = "NX-AI/TiRex"
MIN_CONTEXT_LENGTH = 12


class FixtureNotFoundError(KeyError):
    """Raised by TirexFixtureBackend when no committed fixture matches the
    requested (series, horizon) call. Fail-loud per spec hermetic-CI rule."""


class TirexBackend(Protocol):
    def forecast(self, series: np.ndarray, horizon: int) -> np.ndarray:
        """Return (horizon, 9) quantile predictions."""
        ...


def _series_key(series: np.ndarray, horizon: int) -> str:
    """Stable hash of (rounded series, horizon) for fixture lookup."""
    rounded = np.round(series, 6).astype(np.float32).tobytes()
    return hashlib.sha256(rounded + horizon.to_bytes(4, "big")).hexdigest()[:16]


@dataclass
class TirexFixtureBackend:
    """Backend that replays JSON fixtures from a directory.

    Each fixture: {"key": str, "horizon": int, "quantiles": list[list[float]]}.
    """

    fixture_dir: Path

    def forecast(self, series: np.ndarray, horizon: int) -> np.ndarray:
        key = _series_key(series, horizon)
        path = self.fixture_dir / f"{key}.json"
        if not path.exists():
            raise FixtureNotFoundError(
                f"no TiRex fixture for key={key} (series_len={len(series)}, "
                f"horizon={horizon}). Fixture dir: {self.fixture_dir}. "
                f"Run scripts.regenerate_tirex_fixtures to produce it."
            )
        with path.open() as f:
            data = json.load(f)
        return np.asarray(data["quantiles"], dtype=np.float64)


@dataclass
class TirexHuggingFaceBackend:
    """Live HuggingFace TiRex backend. Loaded lazily on first call.

    Apple Silicon note: LightGBM and PyTorch both link libomp.dylib; running
    both in the same process can deadlock at OpenMP barriers. Set
    OMP_NUM_THREADS=1 and call torch.set_num_threads(1) before instantiating
    the backend, or run TiRex inference in a subprocess. The fixture backend
    is unaffected since it does not import torch.
    """

    model_name: str = TIREX_MODEL_NAME
    device: str | None = None  # "cpu", "cuda", "mps", or None for auto.
    _model: object = None

    def forecast(self, series: np.ndarray, horizon: int) -> np.ndarray:
        if self._model is None:
            try:
                from tirex import load_model
            except ImportError as e:
                raise ImportError(
                    "tirex-ts package not installed; install via "
                    "`uv pip install tirex-ts` for live forecasting"
                ) from e
            self._model = load_model(self.model_name, device=self.device)
        # tirex-ts >= 1.4.1 API: forecast(context, prediction_length) returns
        # (quantiles, mean) where quantiles has shape (B, H, 9) at the default
        # 0.1..0.9 quantile grid that matches QUANTILE_LEVELS.
        import torch

        ctx = torch.from_numpy(np.asarray(series, dtype=np.float32)).unsqueeze(0)
        quantiles, _mean = self._model.forecast(  # type: ignore[attr-defined]
            context=ctx,
            prediction_length=horizon,
        )
        return np.asarray(quantiles[0].detach().cpu().numpy(), dtype=np.float64)


@dataclass
class TirexForecaster:
    """TiRex zero-shot forecaster.

    Owns winsorization (MAD-based outlier cap) before backend call and
    monotonic-quantile enforcement after. Raises ValueError if the input
    series has fewer than MIN_CONTEXT_LENGTH non-NaN points.
    """

    backend: TirexBackend
    outlier_cap: float = ROBUST_OUTLIER_CAP_DEFAULT

    def predict(self, series: np.ndarray, *, horizon: int) -> TirexOutput:
        series = np.asarray(series, dtype=np.float64)
        finite_count = int(np.count_nonzero(~np.isnan(series)))
        if finite_count < MIN_CONTEXT_LENGTH:
            raise ValueError(
                f"series has {finite_count} finite points; TiRex requires "
                f"at least {MIN_CONTEXT_LENGTH}"
            )
        clipped = winsorize_mad(series, cap=self.outlier_cap)
        quantiles = self.backend.forecast(clipped, horizon)
        if quantiles.shape != (horizon, 9):
            raise ValueError(f"backend returned shape {quantiles.shape}; expected ({horizon}, 9)")
        # Enforce monotonic by sorting each horizon row ascending.
        sorted_q = np.sort(quantiles, axis=1)
        return TirexOutput(quantiles=sorted_q, horizon=horizon)
