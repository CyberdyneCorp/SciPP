# integrate (delta)

## ADDED Requirements

### Requirement: Boundary value problem solver
The system SHALL provide `solve_bvp` that solves a first-order ODE system
`y'(x) = fun(x, y)` on a fixed mesh subject to a two-point boundary condition
`bc(y(x[0]), y(x[-1])) = 0`, using 4th-order collocation with a global Newton
solve of the residual system, and returns the solution sampled at the mesh nodes
together with a success flag and message.

#### Scenario: Harmonic oscillator
- GIVEN `y0' = y1`, `y1' = -y0` on `[0, pi/2]` with `bc: y0(0)=0, y0(pi/2)=1`
- WHEN `solve_bvp` is called with a uniform mesh and a crude initial guess
- THEN `success` SHALL be true
- AND the returned `y0` SHALL approximate `sin x` and `y1` SHALL approximate
  `cos x` at the mesh nodes
- AND the nodal values SHALL match `scipy.integrate.solve_bvp` to `allclose`
  tolerance.

#### Scenario: Polynomial source term
- GIVEN `y0' = y1`, `y1' = 6x` on `[0, 1]` with `bc: y0(0)=0, y0(1)=1`
- WHEN `solve_bvp` is called
- THEN the returned `y0` SHALL reproduce the analytic cubic `x^3` and `y1` SHALL
  reproduce `3 x^2` to near machine precision, since 4th-order collocation is
  exact through cubics.

#### Scenario: Invalid inputs
- GIVEN a mesh with fewer than two nodes, or a guess whose column count differs
  from the mesh length
- WHEN `solve_bvp` is called
- THEN it SHALL raise a `value_error`.
