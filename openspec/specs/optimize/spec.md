# optimize Specification

## Purpose
TBD - created by archiving change add-optimize. Update Purpose after archive.
## Requirements
### Requirement: Scalar root finding

`scypp::optimize` SHALL provide `brentq`, `bisect` and `newton` for finding a root
of a scalar function, matching SciPy within documented tolerance. `brentq`/`bisect`
SHALL require a sign-changing bracket and raise an error otherwise. (oracle:
scipy/optimize/_zeros_py.py)

#### Scenario: Bracketing root finders converge
- GIVEN a continuous function with a sign change on `[a, b]`
- WHEN `brentq(f, a, b)` or `bisect(f, a, b)` is called
- THEN it returns a root `x` with `|f(x)|` near zero, `allclose` to SciPy's root

#### Scenario: Newton with and without a derivative
- GIVEN a function (and optionally its derivative) and a starting point
- WHEN `newton(f, x0)` is called
- THEN it converges to a root matching SciPy's `newton`, using the secant method
  when no derivative is supplied

#### Scenario: Invalid bracket
- GIVEN `f(a)` and `f(b)` with the same sign
- WHEN `brentq(f, a, b)` is called
- THEN an error is raised

### Requirement: Scalar minimization

`scypp::optimize` SHALL provide `minimize_scalar` with methods `"brent"` and
`"bounded"`, matching SciPy within documented tolerance. (oracle:
scipy/optimize/_optimize.py)

#### Scenario: Unbounded scalar minimum
- GIVEN a unimodal scalar function
- WHEN `minimize_scalar(f, method="brent")` is called
- THEN the returned minimizer and minimum value are `allclose` to SciPy's

#### Scenario: Bounded scalar minimum
- GIVEN a function and an interval `[a, b]`
- WHEN `minimize_scalar(f, method="bounded", bounds=(a,b))` is called
- THEN the minimizer lies in `[a, b]` and matches SciPy's

### Requirement: Multivariate minimization

`scypp::optimize` SHALL provide `minimize` with methods `"Nelder-Mead"` and
`"BFGS"`, returning an `OptimizeResult` with `x`, `fun`, `success`, `nit` and
`nfev`. Non-convergence SHALL be reported via `success=false`, not an exception.
(oracle: scipy/optimize/_optimize.py)

#### Scenario: Minimize a multivariate function
- GIVEN the Rosenbrock function and a starting point
- WHEN `minimize(f, x0, method="Nelder-Mead")` and `method="BFGS"` are called
- THEN each returns `success=true` and an optimum `x` `allclose` to SciPy's (and to
  the known minimum), with the gradient near zero at the BFGS optimum

#### Scenario: Quadratic bowl
- GIVEN a positive-definite quadratic and any start
- WHEN `minimize(f, x0, method="BFGS")` is called
- THEN it converges to the analytic minimum within tolerance

### Requirement: Nonlinear least squares and curve fitting

`scypp::optimize` SHALL provide `least_squares` (Levenberg–Marquardt),
`curve_fit` (model fitting with parameter covariance) and `fsolve` (multivariate
root finding), matching SciPy within documented tolerance. (oracle:
scipy/optimize/_lsq, _minpack_py.py)

#### Scenario: Fit a model with curve_fit
- GIVEN noisy-free samples of an exponential-decay model and an initial guess
- WHEN `curve_fit(model, xdata, ydata, p0)` is called
- THEN the fitted parameters are `allclose` to the true parameters and to SciPy's
  `popt`, and `pcov` has the expected shape

#### Scenario: least_squares minimizes the residual
- GIVEN a residual function and a start
- WHEN `least_squares(residual, x0)` is called
- THEN the returned `x` drives the residual near zero and matches SciPy's solution

#### Scenario: fsolve finds a vector root
- GIVEN a system of nonlinear equations and a starting point
- WHEN `fsolve(F, x0)` is called
- THEN `F(x)` is near zero at the returned `x`, matching SciPy

### Requirement: Linear programming
The system SHALL provide `linprog(c, A_ub, b_ub, A_eq, b_eq)` that minimizes the
linear objective `c·x` subject to `A_ub·x <= b_ub`, `A_eq·x = b_eq`, and `x >= 0`,
returning the optimal vector, objective value, success flag, and a status code
(0 optimal, 2 infeasible, 3 unbounded).

#### Scenario: Bounded optimum
- GIVEN `c = [-1, -2]`, `A_ub = [[1,1],[1,3]]`, `b_ub = [4, 6]`
- WHEN `linprog` is called
- THEN the result SHALL be `x = [3, 1]` with `fun = -5` and `success = true`
- AND the result SHALL match `scipy.optimize.linprog` to `allclose` tolerance.

#### Scenario: Equality and inequality constraints
- GIVEN `c = [1, 1]`, `A_ub = [[-1,-2]]`, `b_ub = [-3]`, `A_eq = [[1,1]]`, `b_eq = [2]`
- WHEN `linprog` is called
- THEN the optimal objective SHALL equal SciPy's to `allclose` tolerance.

#### Scenario: Infeasible and unbounded problems
- GIVEN constraints with no feasible point
- THEN `success` SHALL be false and `status` SHALL be 2
- AND an unbounded objective SHALL yield `status` 3.

### Requirement: Nonnegative least squares
The system SHALL provide `nnls(A, b)` that solves `min ||A·x - b||` subject to
`x >= 0` via the Lawson-Hanson active-set algorithm, returning the solution and
its residual norm.

#### Scenario: Active constraint
- GIVEN an `A`, `b` whose unconstrained least-squares solution has a negative
  component
- WHEN `nnls` is called
- THEN the returned `x` SHALL be elementwise nonnegative
- AND `x` and `rnorm` SHALL match `scipy.optimize.nnls` to `allclose` tolerance.

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

