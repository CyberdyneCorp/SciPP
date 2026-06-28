# integrate Specification

## Purpose
TBD - created by archiving change add-integrate. Update Purpose after archive.
## Requirements
### Requirement: Fixed-sample quadrature

`scypp::integrate` SHALL provide `trapezoid`, `simpson` and `cumulative_trapezoid`
over a sampled `y` array with optional sample points `x` or uniform spacing `dx`,
matching SciPy. (oracle: scipy/integrate/_quadrature.py)

#### Scenario: Trapezoid and Simpson match SciPy
- GIVEN samples of a function on a grid
- WHEN `trapezoid(y, x)` and `simpson(y, x)` are computed
- THEN the results are `allclose` to SciPy's

#### Scenario: Cumulative trapezoid
- GIVEN samples `y` and spacing `dx`
- WHEN `cumulative_trapezoid(y, dx=dx, initial=0)` is computed
- THEN the result is `allclose` to SciPy and its last element equals
  `trapezoid(y, dx=dx)`

### Requirement: Adaptive and fixed-order quadrature

`scypp::integrate` SHALL provide `quad` (adaptive Gauss–Kronrod returning a value
and error estimate) and `fixed_quad` (Gauss–Legendre of given order), matching
SciPy within the requested tolerance. (oracle: scipy/integrate/_quadpack_py.py)

#### Scenario: Adaptive quad matches SciPy and the analytic integral
- GIVEN a smooth integrand on `[a, b]`
- WHEN `quad(f, a, b)` is called
- THEN the returned value is `allclose` to `scipy.integrate.quad` and to the
  analytic integral, and the reported error estimate bounds the true error

#### Scenario: Fixed-order Gauss–Legendre
- GIVEN a polynomial of degree ≤ 2n−1
- WHEN `fixed_quad(f, a, b, n)` is called
- THEN it integrates the polynomial exactly (within rounding), matching SciPy

### Requirement: Initial-value ODE integration

`scypp::integrate` SHALL provide `solve_ivp` with the explicit Runge–Kutta methods
`"RK45"` and `"RK23"`, adaptive step-size control, `rtol`/`atol` tolerances, and
evaluation at requested `t_eval` points, returning an `OdeResult` with `t`, `y`
(shaped `n_states × n_times`), and `success`. (oracle: scipy/integrate/_ivp)

#### Scenario: Integrate a scalar ODE
- GIVEN `y' = -y`, `y(0) = 1` over `[0, 5]`
- WHEN `solve_ivp(f, (0,5), y0, method="RK45", t_eval=…)` is called
- THEN `y(t)` is `allclose` to `exp(-t)` at the evaluation points and `success` is
  true

#### Scenario: Integrate a system
- GIVEN the harmonic oscillator `y'' = -y` as a first-order system
- WHEN `solve_ivp` integrates it with `"RK45"` and `"RK23"`
- THEN the trajectory matches the analytic `cos`/`sin` solution within tolerance

#### Scenario: Tolerances tighten the solution
- GIVEN a nonlinear ODE with a known solution
- WHEN integrated with smaller `rtol`/`atol`
- THEN the error against the analytic solution decreases

### Requirement: Finite-difference differentiation

`scypp::differentiate` SHALL provide `derivative` (scalar, central differences with
Richardson extrapolation), `jacobian` and `hessian`, matching closed-form
derivatives within documented tolerance. (oracle: scipy/differentiate)

#### Scenario: Scalar derivative
- GIVEN a smooth function and a point
- WHEN `derivative(f, x)` is called
- THEN the returned `df` is `allclose` to the analytic derivative

#### Scenario: Jacobian and Hessian
- GIVEN a vector field and a scalar field
- WHEN `jacobian(F, x)` and `hessian(f, x)` are computed
- THEN they are `allclose` to the analytic Jacobian and Hessian

### Requirement: Stiff ODE integration via Radau
The system SHALL support `solve_ivp` with `method="Radau"`, an implicit Radau IIA
(3-stage, order 5) integrator with adaptive step control, that remains stable on
stiff initial-value problems and returns the solution at the requested evaluation
points.

#### Scenario: Stiff scalar decay
- GIVEN `y' = -20 y`, `y(0) = 1`
- WHEN `solve_ivp` is called with `method="Radau"` and tight tolerances
- THEN the solution SHALL approximate `exp(-20 t)` at the evaluation points
- AND SHALL match `scipy.integrate.solve_ivp(method="Radau")` to `allclose`
  tolerance.

#### Scenario: Stiff linear system
- GIVEN `y1' = -100 y1 + y2`, `y2' = -y2`, `y0 = [1, 1]`
- WHEN `solve_ivp` is called with `method="Radau"`
- THEN the returned trajectory SHALL match SciPy's Radau solution to `allclose`
  tolerance without the step collapse that the explicit methods suffer.

### Requirement: Nested and extended quadrature
The system SHALL provide `romberg`, `quad_vec`, `dblquad`, `tplquad`, and `nquad`,
composing the adaptive `quad` to integrate vector-valued and multidimensional
integrands (including variable inner bounds) to the requested tolerance.

#### Scenario: Romberg integration
- GIVEN `f(x) = sin(x)` on `[0, π]`
- WHEN `romberg` is called
- THEN the result SHALL equal `2` to `allclose` tolerance.

#### Scenario: Vector-valued quadrature
- GIVEN `f(x) = [1, x, x², sin x]` on `[0, 1]`
- WHEN `quad_vec` is called
- THEN each component SHALL match `scipy.integrate.quad_vec` to `allclose`
  tolerance.

#### Scenario: Double integral with variable bounds
- GIVEN `func(y, x) = x + y`, `x ∈ [0, 1]`, `y ∈ [0, x]`
- WHEN `dblquad` is called
- THEN the result SHALL match `scipy.integrate.dblquad` to `allclose` tolerance.

#### Scenario: Triple and N-dimensional integrals
- GIVEN a separable integrand over a hyper-rectangle
- WHEN `tplquad` and `nquad` are called
- THEN both SHALL return the same value, matching SciPy to `allclose` tolerance.

