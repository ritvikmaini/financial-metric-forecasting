# Backlog

- SNOW FY2021 (+364d) and FY2023 (+363d) Q4 still year-lagged after L-INFRA-012 + L-INFRA-013 fixes. Not L-INFRA-012 (calendar-shifted but duration gate handles it) or L-INFRA-013 (no Q3-bucket phantom in this case). Suspect a SNOW-specific normalize edge case in the comparative-period emission. Recency guard in `compute_revenue_ttm` returns None for these as_ofs, so no feature corruption; investigate before S5.
