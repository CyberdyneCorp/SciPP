# stats Specification

## Purpose
TBD - created by archiving change add-stats. Update Purpose after archive.
## Requirements
### Requirement: Continuous probability distributions

`scypp::stats` SHALL provide the continuous distributions `norm`, `expon`,
`uniform`, `gamma`, `chi2`, `beta`, `t` and `f`, each exposing `pdf`, `logpdf`,
`cdf`, `sf`, `ppf` and the moments `mean`/`var`/`std`/`median`, parameterized by
`loc`/`scale` (and shape parameters), matching SciPy within documented tolerance.
(oracle: scipy/stats/_continuous_distns.py)

#### Scenario: pdf/cdf match SciPy
- GIVEN a distribution with given parameters and a set of points
- WHEN `pdf` and `cdf` are evaluated
- THEN the results are `allclose` to `scipy.stats`'s for that distribution

#### Scenario: ppf inverts cdf
- GIVEN a distribution and probabilities `q` in `(0,1)`
- WHEN `ppf(q)` is computed
- THEN `cdf(ppf(q))` is `allclose` to `q`, and `ppf` matches SciPy

#### Scenario: Survival and moments
- GIVEN a distribution
- WHEN `sf(x)` and `mean`/`var` are evaluated
- THEN `sf(x)` is `allclose` to `1 − cdf(x)` and the moments match SciPy

### Requirement: Summary statistics

`scypp::stats` SHALL provide `gmean`, `hmean`, `describe`, `moment`, `skew`,
`kurtosis`, `sem`, `variation`, `zscore`, `iqr`, `rankdata` and `mode`, matching
SciPy. (oracle: scipy/stats/_stats_py.py)

#### Scenario: Descriptive statistics match SciPy
- GIVEN a sample
- WHEN `gmean`, `hmean`, `skew`, `kurtosis`, `sem`, `variation` and `iqr` are
  computed
- THEN each is `allclose` to SciPy's value

#### Scenario: Ranks and z-scores
- GIVEN a sample (with ties)
- WHEN `rankdata` and `zscore` are computed
- THEN `rankdata` assigns average ranks to ties and `zscore` is `allclose` to
  SciPy

### Requirement: Correlation and regression

`scypp::stats` SHALL provide `pearsonr`, `spearmanr` and `linregress`, returning
the coefficient/slope and associated p-value/statistics, matching SciPy. (oracle:
scipy/stats/_stats_py.py)

#### Scenario: Correlation coefficients and p-values
- GIVEN paired samples
- WHEN `pearsonr` and `spearmanr` are computed
- THEN the correlation and two-sided p-value are `allclose` to SciPy's

#### Scenario: Linear regression
- GIVEN paired samples
- WHEN `linregress(x, y)` is computed
- THEN slope, intercept, rvalue, pvalue and stderr are `allclose` to SciPy's

### Requirement: Parametric hypothesis tests

`scypp::stats` SHALL provide `ttest_1samp`, `ttest_ind`, `ttest_rel`, `f_oneway`,
`ks_2samp`, `chi2_contingency` and `normaltest`, each returning a statistic and a
p-value matching SciPy within documented tolerance. (oracle: scipy/stats/_stats_py.py)

#### Scenario: t-tests match SciPy
- GIVEN one or two samples
- WHEN `ttest_1samp`, `ttest_ind` and `ttest_rel` are computed
- THEN the t statistic and p-value are `allclose` to SciPy's

#### Scenario: ANOVA, KS and contingency tests
- GIVEN the appropriate inputs (groups / two samples / a contingency table)
- WHEN `f_oneway`, `ks_2samp` and `chi2_contingency` are computed
- THEN the statistics and p-values are `allclose` to SciPy's

#### Scenario: Normality test
- GIVEN a sample
- WHEN `normaltest` is computed
- THEN the K² statistic and p-value are `allclose` to SciPy's

### Requirement: Gaussian kernel density estimation

`scypp::stats` SHALL provide `gaussian_kde` with Scott and Silverman bandwidth
selection, evaluating the estimated density at query points, matching SciPy.
(oracle: scipy/stats/_kde.py)

#### Scenario: KDE matches SciPy
- GIVEN a sample and query points
- WHEN a `gaussian_kde` is constructed and evaluated at the query points
- THEN the density estimates are `allclose` to `scipy.stats.gaussian_kde`, and the
  density integrates to approximately 1

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

