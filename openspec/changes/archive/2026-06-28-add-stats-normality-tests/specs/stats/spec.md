# stats (delta)

## ADDED Requirements

### Requirement: Normality tests (Shapiro-Wilk and Anderson-Darling)
The system SHALL provide `scypp::stats::shapiro`, computing the Shapiro-Wilk W
statistic and its p-value via Royston's AS R94 algorithm, and
`scypp::stats::anderson`, computing the Anderson-Darling A^2 statistic for the
normal distribution together with SciPy's critical values and significance
levels, both matching `scipy.stats` to oracle tolerance.

#### Scenario: Shapiro-Wilk on a roughly normal sample
- GIVEN a 20-element sample drawn from an approximately normal distribution
- WHEN `shapiro(x)` is called
- THEN the returned `statistic` SHALL match `scipy.stats.shapiro(x).statistic`
  to `1e-4`
- AND the returned `pvalue` SHALL match `scipy.stats.shapiro(x).pvalue` to `1e-3`.

#### Scenario: Shapiro-Wilk on a skewed sample and the n==3 exact branch
- GIVEN a skewed sample and, separately, a 3-element sample
- WHEN `shapiro(x)` is called
- THEN the `statistic` and `pvalue` SHALL match SciPy to `1e-4` and `1e-3`
- AND the `n == 3` case SHALL use the exact arcsine p-value formula.

#### Scenario: Shapiro-Wilk rejects fewer than three observations
- GIVEN a sample with fewer than three observations
- WHEN `shapiro(x)` is called
- THEN it SHALL throw `std::invalid_argument`.

#### Scenario: Anderson-Darling for the normal distribution
- GIVEN a sample standardised with the sample mean and std (`ddof=1`)
- WHEN `anderson(x)` is called
- THEN the returned `statistic` SHALL match `scipy.stats.anderson(x,'norm').statistic`
  to `1e-4`
- AND `critical_values` SHALL equal SciPy's values for significance levels
  `[15, 10, 5, 2.5, 1]%` exactly
- AND `significance_level` SHALL equal `[15, 10, 5, 2.5, 1]`.
