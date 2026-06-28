# Add Shapiro-Wilk and Anderson-Darling normality tests

## Why

`scipp::stats` already ships parametric and rank tests plus `normaltest`
(D'Agostino-Pearson), but lacks the two most widely used dedicated normality
tests: Shapiro-Wilk (`scipy.stats.shapiro`) and Anderson-Darling
(`scipy.stats.anderson` for the normal distribution). They were deferred from
the `add-stats-extras` backlog. Neither needs NumPP changes — both reuse the
existing `norm` distribution helpers — so they are a clean drawdown.

## What changes

Extends the **stats** capability, validated against the SciPy 1.15 oracle:

- **`shapiro(x)` -> `ShapiroResult{statistic, pvalue}`**: Royston's AS R94
  algorithm — expected normal order-statistic weights, the W statistic computed
  as a squared correlation, and Royston's polynomial p-value approximation
  (with the exact arcsine p-value for `n == 3`). Valid `n` in `[3, ~5000]`.
- **`anderson(x)` -> `AndersonResult{statistic, critical_values, significance_level}`**:
  Anderson-Darling A^2 for the normal case, standardised with the sample mean
  and std (`ddof=1`) exactly as SciPy does, returning SciPy's critical values
  for significance levels `[15, 10, 5, 2.5, 1]%`.

## Impact

- Affected specs: **modifies** the `stats` capability (adds one requirement).
- Affected code: new `src/stats/normality.cpp`, declarations and result structs
  in `include/scipp/stats/stats.hpp`, `tests/test_stats_normality.cpp`, extended
  oracle generator. Wired into `src/CMakeLists.txt` and `tests/CMakeLists.txt`.
- Trims the shapiro/anderson item from the `add-stats-extras` backlog.

## Non-goals

- Anderson-Darling for distributions other than `norm` (expon, logistic,
  gumbel) and `anderson_ksamp` — remain in the backlog.
- The small-sample correction is applied to the **critical values**
  (`_Avals / (1 + 4/N - 25/N^2)`) rather than to the statistic: this matches
  the current SciPy 1.15 oracle, which returns the raw A^2 statistic.
