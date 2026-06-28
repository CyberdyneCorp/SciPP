# integrate (delta)

## ADDED Requirements

### Requirement: Stiff ODE integration via BDF
The system SHALL support `solve_ivp` with `method="BDF"`, an implicit backward
differentiation integrator of variable order 1–2 with adaptive step control, that
remains stable on stiff initial-value problems and returns the solution at the
requested evaluation points.

#### Scenario: Stiff scalar decay
- GIVEN `y' = -20 y`, `y(0) = 1`
- WHEN `solve_ivp` is called with `method="BDF"` and tight tolerances
- THEN the solution SHALL approximate the analytic `exp(-20 t)` at the evaluation
  points to `allclose` tolerance (~1e-5).

#### Scenario: Stiff linear system
- GIVEN `y1' = -100 y1 + y2`, `y2' = -y2`, `y0 = [1, 1]`
- WHEN `solve_ivp` is called with `method="BDF"`
- THEN the returned trajectory SHALL match the analytic solution
  `y2 = exp(-t)`, `y1 = (1 - 1/99) exp(-100 t) + (1/99) exp(-t)` to `allclose`
  tolerance (~1e-5) without the step collapse that the explicit methods suffer.
