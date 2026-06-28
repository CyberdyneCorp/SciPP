# integrate (delta)

## ADDED Requirements

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
