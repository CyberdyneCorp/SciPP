# optimize Specification

## ADDED Requirements

### Requirement: Linear and mixed-integer programming
`scypp::optimize` SHALL provide `linprog` (linear programming) and `milp`
(mixed-integer linear programming), matching SciPy within tolerance. (oracle: scipy/optimize/_linprog.py, _milp.py)

#### Scenario: linprog finds the optimum
- GIVEN a linear objective with linear inequality/equality constraints and bounds
- WHEN `linprog` is called
- THEN the returned optimum and objective value match SciPy and satisfy the constraints

### Requirement: Additional and constrained minimize methods
`scypp::optimize::minimize` SHALL provide the methods `Powell`, `CG`, `Newton-CG`,
`L-BFGS-B`, `SLSQP`, `trust-constr` and `COBYLA`, honoring bound and constraint
specifications, matching SciPy. (oracle: scipy/optimize/_minimize.py)

#### Scenario: Constrained minimum
- GIVEN an objective with bounds and (non)linear constraints
- WHEN `minimize(method="SLSQP")` (and `trust-constr`) is called
- THEN the optimum satisfies the constraints and matches SciPy

### Requirement: Global optimizers
`scypp::optimize` SHALL provide `differential_evolution`, `basinhopping`,
`dual_annealing` and `shgo`, matching SciPy distributionally/within tolerance on
benchmark problems. (oracle: scipy/optimize/_differentialevolution.py, _basinhopping.py)

#### Scenario: Global optimizer finds the global minimum
- GIVEN a multimodal benchmark (e.g. Rastrigin) and a seed
- WHEN a global optimizer runs
- THEN it locates the known global minimum within tolerance

### Requirement: Non-negative and linear least squares
`scypp::optimize` SHALL provide `nnls` and `lsq_linear`, matching SciPy. (oracle: scipy/optimize/_nnls.py, _lsq)

#### Scenario: nnls returns a non-negative solution
- GIVEN `A` and `b`
- WHEN `nnls(A, b)` is called
- THEN the solution is non-negative, minimizes `‖Ax − b‖`, and matches SciPy
