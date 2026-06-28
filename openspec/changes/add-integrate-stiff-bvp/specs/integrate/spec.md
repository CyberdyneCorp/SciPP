# integrate Specification

## ADDED Requirements

### Requirement: Implicit stiff ODE solvers

`scipp::integrate::solve_ivp` SHALL provide the implicit methods `"Radau"`,
`"BDF"` and `"LSODA"` for stiff initial-value problems, each using a Jacobian
(finite-difference by default, or user-supplied) and a Newton iteration per step,
matching SciPy within documented tolerance. A legacy `odeint` entry point SHALL
also be provided. (oracle: scipy/integrate/_ivp, scipy/integrate/odepack)

#### Scenario: Stiff problem integrated accurately
- GIVEN a stiff initial-value problem (e.g. the Van der Pol oscillator at large μ,
  or the Robertson kinetics system)
- WHEN `solve_ivp(f, t_span, y0, method="BDF")` (and `"Radau"`, `"LSODA"`) is called
- THEN the trajectory at the evaluation points is `allclose` to SciPy's, where an
  explicit RK method would require prohibitively many steps

#### Scenario: User-supplied Jacobian is honored
- GIVEN a stiff problem with an analytic Jacobian
- WHEN it is passed to a stiff method
- THEN the solver uses it and converges in fewer function evaluations than with the
  finite-difference Jacobian

### Requirement: Boundary-value problem solver

`scipp::integrate` SHALL provide `solve_bvp` for two-point boundary-value problems
via collocation with a Newton residual solve and mesh refinement, matching SciPy
within documented tolerance. (oracle: scipy/integrate/_bvp.py)

#### Scenario: BVP with known solution
- GIVEN a two-point boundary-value problem with a known analytic solution
- WHEN `solve_bvp(fun, bc, x, y_guess)` is called
- THEN the returned solution sampled on the mesh is `allclose` to the analytic
  solution and `success` is true

### Requirement: Multidimensional and additional quadrature

`scipp::integrate` SHALL provide `dblquad`, `tplquad` and `nquad` (nested adaptive
quadrature) and `romberg`, matching SciPy within the requested tolerance. (oracle:
scipy/integrate/_quadpack_py.py)

#### Scenario: Double integral matches SciPy
- GIVEN a smooth integrand over a rectangular or variable-limit 2-D region
- WHEN `dblquad(f, a, b, gfun, hfun)` is called
- THEN the result is `allclose` to `scipy.integrate.dblquad` and to the analytic
  value

#### Scenario: Romberg integration converges
- GIVEN a smooth 1-D integrand
- WHEN `romberg(f, a, b)` is called
- THEN the result is `allclose` to the analytic integral
