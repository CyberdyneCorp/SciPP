# Design — stats (Phase 7)

## Context

`scipp::stats` ports the deterministic core of `scipy.stats`. Distribution CDFs
reduce to the regularized incomplete gamma/beta integrals, so this change first
implements those special-function enablers (which Phase 1 did not cover) and then
builds the distributions, statistics, tests, and KDE on top. Everything is
deterministic, so results match SciPy to high precision.

## Special-function enablers (internal)

- **gammainc(a, x)** — regularized lower incomplete gamma `P(a,x)`: power series
  for `x < a+1`, Lentz continued fraction for `x ≥ a+1` (Numerical Recipes `gammp`).
  `gammaincc = 1 − gammainc`.
- **betainc(a, b, x)** — regularized incomplete beta `I_x(a,b)`: the standard
  continued fraction with the `x > (a+1)/(a+b+2)` symmetry reflection.
- **gammaincinv / betaincinv** — invert via Newton steps with bisection
  safeguards (used by the `ppf` of gamma/chi2/beta/t/f).

These match `scipy.special.gammainc`/`betainc` and underpin the distributions.

## Distribution API

Each distribution is a struct of static methods mirroring SciPy's
`dist.method(x, *shape, loc=0, scale=1)`:

```cpp
namespace scipp::stats {
struct norm {
  static double pdf(double x, double loc = 0, double scale = 1);
  static double logpdf(double x, double loc = 0, double scale = 1);
  static double cdf(double x, double loc = 0, double scale = 1);
  static double sf (double x, double loc = 0, double scale = 1);
  static double ppf(double q, double loc = 0, double scale = 1);
  static double mean(double loc = 0, double scale = 1);
  static double var (double loc = 0, double scale = 1);
  static double std (double loc = 0, double scale = 1);
  static double median(double loc = 0, double scale = 1);
  static ndarray pdf(const ndarray& x, double loc = 0, double scale = 1);   // elementwise
  static ndarray cdf(const ndarray& x, double loc = 0, double scale = 1);
};
// gamma/beta/t/f/chi2/expon/uniform/lognorm follow, shape params first:
//   gamma::pdf(x, a, loc=0, scale=1);  t::cdf(x, df, loc=0, scale=1); ...
}
```

Standardization: each method maps to the standard form via `z = (x − loc)/scale`
(`pdf` divides by `scale`, `ppf` multiplies and adds `loc`). The standard forms:

- **norm** — `erf`-based `cdf`; `ppf` via `erfinv`.
- **expon/uniform** — closed form.
- **gamma(a)** — `cdf = gammainc(a, z)`, `pdf = z^{a−1} e^{−z}/Γ(a)`; `ppf` via
  `gammaincinv`. **chi2(df) = gamma(df/2, scale=2)**.
- **beta(a,b)** — `cdf = betainc(a, b, z)`; `ppf` via `betaincinv`.
- **t(df)** — `cdf` from `betainc(df/2, ½, df/(df+t²))` with the sign split.
- **f(dfn,dfd)** — `cdf` from `betainc` on `dfn·x/(dfn·x+dfd)`.
- **lognorm(s)** — `norm` of `log`.

## Statistics, correlation, tests

- **Summary**: `gmean`/`hmean` (log/reciprocal means), `moment` (central),
  `skew`/`kurtosis` (Fisher, with SciPy's bias default), `sem`, `variation`,
  `zscore` (ddof default 0), `iqr`, `rankdata` (average ties), `mode`, `describe`.
- **Correlation/regression**: `pearsonr` (r + two-sided p via `t`), `spearmanr`
  (Pearson on ranks), `linregress` (slope/intercept/r/p/stderr).
- **Tests** (statistic + p-value, matching SciPy):
  `ttest_1samp`/`ttest_ind`(equal-var)/`ttest_rel` via the `t` distribution;
  `f_oneway` via `f`; `chi2_contingency` via `chi2`; `ks_2samp` via the
  Kolmogorov asymptotic; `normaltest` (D'Agostino–Pearson K²) via `skewtest` +
  `kurtosistest`.
- Result objects are small structs (e.g. `TtestResult{statistic, pvalue}`).

## gaussian_kde

Bandwidth via Scott (`n^{−1/(d+4)}`) or Silverman; covariance from the data
(`numpp` covariance), factor² scales it; `evaluate(points)` sums the multivariate
normal contributions. Matches `scipy.stats.gaussian_kde` for a given dataset.

## Oracle strategy

Distributions are checked at representative points against `scipy.stats`
(`pdf`/`cdf`/`ppf` round-trip `cdf(ppf(q)) ≈ q`); statistics and tests compare
`statistic`/`pvalue` to SciPy; KDE compares `evaluate` at query points. Tolerances
are tight (`~1e-9`) for closed-form pieces, slightly looser (`~1e-7`) where Newton
inversion or series truncation is involved.

## Open questions

- `skew`/`kurtosis` default to SciPy's biased estimator (`bias=True`); the
  unbiased variants can be added via a flag later. `ttest_ind` implements the
  equal-variance (Student) case; Welch (`equal_var=False`) is a follow-up.
