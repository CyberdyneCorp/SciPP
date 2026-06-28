# optimize Specification

## ADDED Requirements

### Requirement: Scalar root finding

`scipp::optimize` SHALL provide `brentq`, `bisect` and `newton` for finding a root
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

`scipp::optimize` SHALL provide `minimize_scalar` with methods `"brent"` and
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

`scipp::optimize` SHALL provide `minimize` with methods `"Nelder-Mead"` and
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

`scipp::optimize` SHALL provide `least_squares` (Levenberg–Marquardt),
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
