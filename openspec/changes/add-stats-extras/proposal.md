# Stats extras (deferred backlog)

## Why

Phase 7 (`add-stats`, archived) delivered the deterministic distribution/test
core. This change is the **tracked backlog** for sampling, discrete distributions,
QMC, and nonparametric tests. Not implemented yet.

## What changes

Adds (as target requirements) to the **stats** capability:

- **Sampling & fitting**: distribution `rvs` and `fit` (MLE), needing a stats RNG
  stream built on NumPP `random`.
- **Discrete distributions**: `binom`, `poisson`, `geom`, `nbinom`, `hypergeom`,
  `bernoulli`, `randint` (pmf/cdf/ppf/stats).
- **Quasi-Monte-Carlo**: `qmc.Sobol`, `qmc.Halton`, `qmc.LatinHypercube`,
  `qmc.discrepancy`.
- **Nonparametric / rank tests**: `mannwhitneyu`, `wilcoxon`, `kruskal`,
  `kendalltau`, `shapiro`, `anderson`, `friedmanchisquare`.
- **More**: `binned_statistic`, `theilslopes`, `bootstrap`, circular statistics
  (`circmean`/`circstd`), and the long tail of continuous distributions.

## Non-goals
- Implementing anything here; tracking only.
