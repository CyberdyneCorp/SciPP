# Orthogonal distance regression (odr)

## Why

`scipy.odr` (orthogonal distance regression / Deming regression) fits a model when
BOTH variables carry error, minimizing orthogonal distance rather than vertical
residuals. ScyPP had no `odr` capability at all — it was the one whole SciPy module,
besides the trivial `datasets`, that was entirely unscoped. It is moderate value and
builds directly on the existing `scypp::optimize` least-squares machinery; pure CPU,
no NumPP changes.

## What changes

Adds a new **odr** capability — `scypp::odr` — validated against the SciPy oracle:

- A `Model` (the fitting function `f(beta, x)`), a `Data` (x, y with optional
  per-point standard deviations sx, sy), and an `ODR` driver whose `run()` returns
  the estimated parameters, their standard errors, the parameter covariance, the
  residual variance, and the sum of squares — mirroring `scipy.odr.ODR(...).run()`.
- The fit is cast as an augmented Levenberg-Marquardt least-squares problem over
  `[beta, delta]` (the per-point x-offsets), reusing `scypp::optimize::least_squares`.
  Equal weights reduce to ordinary total least squares; per-point sx/sy give the
  weighted ODRPACK fit.
- Oracle tests against `scipy.odr` on a linear model, a nonlinear (`b0·exp(b1·x)`)
  model, and a weighted linear model: `beta` to 1e-5, `res_var`/`sum_square` to 1e-4,
  `sd_beta` loosely.

## Non-goals

- The full ODRPACK feature set (implicit models, multi-response/multidimensional
  responses, fixed-parameter masks beyond the common case, explicit-vs-implicit
  job control, work-array introspection). These can be layered on later.
