#!/usr/bin/env bash
# fmf-public — placeholder reproducer.
#
# The actual entry point lands in S11 (baseline) and gains a --pipeline
# mode in S12 (production-shape pipeline chain). For now this script
# exits with a clear message so a reviewer who clones the repo today
# is not misled about what `--quick` runs.

set -euo pipefail

echo "run_best.sh is a placeholder until S11 lands the baseline benchmark."
echo "See README.md and docs/FORECASTING.md (when published) for status."
exit 0
