# optimize (delta)

## ADDED Requirements

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
