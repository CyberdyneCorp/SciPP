# stats Specification

## ADDED Requirements

### Requirement: Discrete distributions
`scypp::stats` SHALL provide the discrete distributions `binom`, `poisson`,
`geom`, `bernoulli`, `nbinom` and `hypergeom` with `pmf`/`logpmf`/`cdf`/`sf`/`ppf`
and `mean`/`var`, matching SciPy. (oracle: scipy/stats/_discrete_distns.py)

#### Scenario: pmf and cdf match SciPy
- GIVEN a discrete distribution and parameters
- WHEN `pmf` and `cdf` are evaluated at integer support points
- THEN the results are `allclose` to SciPy's

#### Scenario: ppf inverts cdf
- GIVEN a discrete distribution and a probability `q`
- WHEN `ppf(q)` is computed
- THEN `cdf(ppf(q)) >= q` and it matches SciPy's `ppf`

### Requirement: Nonparametric rank tests
`scypp::stats` SHALL provide `mannwhitneyu`, `wilcoxon`, `kruskal` and
`kendalltau`, returning a statistic and an asymptotic p-value (with tie
correction) matching SciPy. (oracle: scipy/stats/_stats_py.py, _mannwhitneyu.py)

#### Scenario: Rank tests match SciPy
- GIVEN appropriate samples
- WHEN `mannwhitneyu`, `wilcoxon` and `kruskal` are computed with the asymptotic
  method
- THEN the statistic and p-value are `allclose` to SciPy's

#### Scenario: Kendall's tau
- GIVEN paired samples (possibly with ties)
- WHEN `kendalltau` is computed
- THEN the tau-b coefficient and p-value are `allclose` to SciPy's
