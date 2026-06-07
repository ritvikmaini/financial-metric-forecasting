#!/usr/bin/env bash
# financial-metric-forecasting reproducer entry.
#
# Without arguments: prints a status note and exits 0.
# With --pipeline: runs the S12 production pipeline chain end-to-end against
# the fixture, building an inference dataset, scoring it through a saved
# LightGBM model, and quality-checking the predictions parquet.

set -euo pipefail

if [[ "${1:-}" == "--pipeline" ]]; then
    exec python scripts/run_pipeline.py chain \
        --as-of 2024-05-15 \
        --feature revenue_ttm \
        --feature gross_margin \
        --feature net_margin \
        --model-path reports/models/lgbm_eps
fi

echo "run_best.sh: pass --pipeline to invoke the S12 pipeline chain."
echo "Without arguments this script is a no-op; see README.md for status."
exit 0
