# Add discrete distributions + rank tests (stats)

## Why

Draws down the highest value-per-effort items from the `add-stats-extras` backlog
that need **nothing from NumPP** — they're deterministic CPU algorithms reusing
SciPP's existing `gammainc`/`betainc` special functions and the Phase-7 continuous
distributions. Discrete distributions and nonparametric rank tests are everyday
statistics tools.

## What changes

Extends the **stats** capability:

- **Discrete distributions**: `binom`, `poisson`, `geom`, `bernoulli`, `nbinom`,
  `hypergeom` with `pmf`/`logpmf`/`cdf`/`sf`/`ppf` and `mean`/`var`.
- **Rank-based tests**: `mannwhitneyu`, `wilcoxon`, `kruskal`, `kendalltau`
  (asymptotic / normal-approximation p-values with tie correction).

## Impact

- Affected specs: extends `stats`.
- Affected code: `src/stats/discrete.cpp`, `src/stats/ranktests.cpp`, header
  additions, `tests/test_stats_discrete.cpp`, extended oracle generator.
- Trims the corresponding items from the open `add-stats-extras` backlog.

## Non-goals

- `rvs`/`fit` sampling and QMC (separate backlog items), `shapiro`/`anderson`
  (special null distributions), and the continuous-distribution long tail.
