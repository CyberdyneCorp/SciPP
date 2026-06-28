# Orthogonal distance regression (odr) — deferred backlog

## Why

`scipy.odr` (orthogonal distance regression / Deming regression) fits a model when
BOTH variables carry error, minimizing orthogonal distance rather than vertical
residuals. ScyPP has no `odr` capability at all — it is the one whole SciPy module,
besides the trivial `datasets`, that is entirely unscoped. It is moderate value and
builds directly on the existing `scypp::optimize` least-squares machinery; pure CPU,
no NumPP changes.

## What changes

Adds a new **odr** capability — `scypp::odr`, validated against the SciPy oracle:

- A `Model` (the fitting function f(beta, x)), `Data` (x, y with optional weights),
  and `ODR` driver that runs the implicit/explicit orthogonal-distance fit and
  returns estimated parameters, standard errors, residual variance, and the sum of
  squares — mirroring `scipy.odr.ODR(...).run()` output fields.

## Non-goals

- The full ODRPACK feature set (implicit models, multi-response, fixed-parameter
  masks beyond the common case, work-array introspection).
- Implementing anything in this change itself; it is a tracking artifact.
