# fmf-public

Financial Metrics Forecasting on public data. Methodology designed at Bavest, reproduced here on SEC EDGAR fundamentals and yfinance prices.

**Status:** Under active development. This is a v0 stub; v1.0 release will carry the headline scoreboard, noise-floor analysis, and dead-CV bug flagship.

## Project layout

```
fmf-public/
├── fmf/                      # the package
│   └── data/                 # DuckDB substrate
├── docs/                     # pillar docs, specs, knowledge ledger
├── reports/                  # figures, scoreboard snapshots, noise_floor.json
├── scripts/                  # CLI entry points
└── tests/                    # mirrors fmf/ tree
```

## Reproducibility (placeholder until S11 baseline)

```bash
git clone https://github.com/<your-handle>/fmf-public.git
cd fmf-public
uv sync --extra dev
bash run_best.sh --quick      # placeholder until S11
```

## License

Code under `MIT`; documentation under `CC-BY-4.0`. See `LICENSE-CODE` and `LICENSE-DOCS`.

## Citation

See `CITATION.cff`.
