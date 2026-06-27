# stats Specification

## ADDED Requirements

### Requirement: Distribution sampling and fitting
`scypp::stats` distributions SHALL provide `rvs` (random sampling, seeded via
NumPP `random`) and `fit` (maximum-likelihood parameter estimation), matching
SciPy reproducibly or distributionally. (oracle: scipy/stats/_continuous_distns.py)

#### Scenario: Seeded sampling and fit recover parameters
- GIVEN a distribution with known parameters
- WHEN a large seeded sample is drawn with `rvs` and refit with `fit`
- THEN the fitted parameters are close to the true parameters and sampling is
  reproducible for a fixed seed

### Requirement: Discrete distributions
`scypp::stats` SHALL provide the discrete distributions `binom`, `poisson`,
`geom`, `nbinom`, `hypergeom` and `bernoulli` with `pmf`/`logpmf`/`cdf`/`ppf` and
moments, matching SciPy. (oracle: scipy/stats/_discrete_distns.py)

#### Scenario: pmf/cdf match SciPy
- GIVEN a discrete distribution and parameters
- WHEN `pmf` and `cdf` are evaluated at integer support points
- THEN the results are `allclose` to SciPy

### Requirement: Quasi-Monte-Carlo
`scypp::stats::qmc` SHALL provide `Sobol`, `Halton` and `LatinHypercube`
low-discrepancy samplers, matching SciPy. (oracle: scipy/stats/_qmc.py)

#### Scenario: Sobol sequence matches SciPy
- GIVEN a dimension and sample count (power of two for Sobol)
- WHEN the sampler generates points
- THEN the points match SciPy's sequence

### Requirement: Nonparametric and rank-based tests
`scypp::stats` SHALL provide `mannwhitneyu`, `wilcoxon`, `kruskal`, `kendalltau`,
`shapiro` and `anderson`, matching SciPy within documented tolerance. (oracle: scipy/stats/_stats_py.py, _morestats.py)

#### Scenario: Rank tests match SciPy
- GIVEN appropriate samples
- WHEN `mannwhitneyu` / `wilcoxon` / `kruskal` are computed
- THEN the statistic and p-value match SciPy
