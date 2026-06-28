# optimize (delta)

## ADDED Requirements

### Requirement: Additional minimize methods
The system SHALL extend `minimize(f, x0, method, tol, maxiter, bounds)` with the
methods `"Powell"`, `"CG"`, and `"L-BFGS-B"`, returning an `OptimizeResult` whose
optimum `x` and objective `fun` match `scipy.optimize.minimize` for the same
method to `allclose` tolerance. `"L-BFGS-B"` SHALL accept an optional `bounds`
argument of per-coordinate `(lo, hi)` pairs and enforce them; existing callers
that omit `bounds` SHALL continue to compile and run.

#### Scenario: Powell on Rosenbrock and a shifted quadratic
- GIVEN the Rosenbrock function started at `[-1.2, 1.0]`
- WHEN `minimize` is called with `method="Powell"`
- THEN `x` SHALL converge to `[1, 1]` and `fun` to `0` matching SciPy
- AND the shifted quadratic `(x-3)^2 + (y+1)^2` SHALL minimize at `[3, -1]`.

#### Scenario: CG with numerical gradient
- GIVEN the Rosenbrock function started at `[-1.2, 1.0]`
- WHEN `minimize` is called with `method="CG"`
- THEN `x` SHALL match SciPy's CG result to `~1e-4` and `fun` to `~1e-6`
- AND the shifted quadratic SHALL minimize at `[3, -1]`.

#### Scenario: L-BFGS-B unbounded
- GIVEN the Rosenbrock function and the shifted quadratic started at the same
  points without bounds
- WHEN `minimize` is called with `method="L-BFGS-B"`
- THEN `x` and `fun` SHALL match SciPy's L-BFGS-B result to `~1e-4` / `~1e-6`.

#### Scenario: L-BFGS-B with box bounds
- GIVEN the shifted quadratic whose unconstrained minimum `[3, -1]` lies outside
  the box `bounds = [(0, 2), (0, 5)]`
- WHEN `minimize` is called with `method="L-BFGS-B"` and those bounds
- THEN the constrained optimum SHALL be `x = [2, 0]` with `fun = 2`, matching
  `scipy.optimize.minimize(method="L-BFGS-B", bounds=...)`.
