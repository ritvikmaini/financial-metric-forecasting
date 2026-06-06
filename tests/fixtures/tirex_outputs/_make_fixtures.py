"""Generate the three committed TiRex fixture JSON files for CI.

Run once (or whenever the fixture series change) to refresh
tests/fixtures/tirex_outputs/*.json. Not a pytest test; just a script.
"""

import json
from pathlib import Path

import numpy as np

from fmf.equity.forecasting.models._tirex_preprocessing import winsorize_mad
from fmf.equity.forecasting.models.tirex_model import _series_key

FIXTURE_DIR = Path(__file__).parent


def _make_fixture(series: np.ndarray, horizon: int, quantiles: np.ndarray) -> None:
    key = _series_key(series, horizon)
    (FIXTURE_DIR / f"{key}.json").write_text(
        json.dumps(
            {
                "key": key,
                "horizon": horizon,
                "series_len": len(series),
                "quantiles": quantiles.tolist(),
            },
            indent=2,
        )
    )


def main() -> None:
    # Fixture 1: constant-growth 24-point series, horizon=4.
    series_normal = np.linspace(100.0, 220.0, 24)
    q_normal = np.tile(np.linspace(220.0, 240.0, 4)[:, None], (1, 9))
    q_normal += np.linspace(-10.0, 10.0, 9)[None, :]  # spread quantiles
    _make_fixture(series_normal, 4, q_normal)

    # Fixture 2: noisy series with outliers (after winsorize).
    rng = np.random.default_rng(0)
    base = np.linspace(100.0, 200.0, 24)
    series_noisy = base + rng.normal(scale=5.0, size=24)
    series_noisy[3] = 1000.0  # outlier
    clipped = winsorize_mad(series_noisy)
    q_noisy = np.tile(np.linspace(200.0, 220.0, 4)[:, None], (1, 9))
    q_noisy += np.linspace(-15.0, 15.0, 9)[None, :]
    _make_fixture(clipped, 4, q_noisy)

    # Fixture 3: minimum context, horizon=2.
    series_min = np.arange(12.0, 24.0)
    q_min = np.tile(np.linspace(24.0, 26.0, 2)[:, None], (1, 9))
    q_min += np.linspace(-2.0, 2.0, 9)[None, :]
    _make_fixture(series_min, 2, q_min)

    print(f"wrote 3 fixtures to {FIXTURE_DIR}")


if __name__ == "__main__":
    main()
