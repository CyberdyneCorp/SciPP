# linalg Specification

## ADDED Requirements

### Requirement: LDL decomposition
`scipp::linalg` SHALL provide `ldl` for symmetric/Hermitian indefinite matrices
(Bunch–Kaufman with 2×2 pivot blocks), matching SciPy. (oracle: scipy/linalg/_decomp_ldl.py)

#### Scenario: LDL reconstructs the input
- GIVEN a symmetric indefinite matrix `A`
- WHEN `ldl(A)` returns `lu, d, perm`
- THEN the permuted `lu @ d @ lu.conj().T` reconstructs `A` and matches SciPy

### Requirement: Matrix functions
`scipp::linalg` SHALL provide `sqrtm`, `logm`, `funm`, `cosm`, `sinm` and
`fractional_matrix_power`, matching SciPy within documented tolerance. (oracle: scipy/linalg/_matfuncs.py)

#### Scenario: Matrix square root and logarithm
- GIVEN a matrix `A` with a principal square root / logarithm
- WHEN `sqrtm(A)` and `logm(A)` are computed
- THEN `sqrtm(A) @ sqrtm(A)` ≈ `A`, `expm(logm(A))` ≈ `A`, and both match SciPy

### Requirement: Schur and generalized decompositions
`scipp::linalg` SHALL provide `schur`, `rsf2csf`, `qz`, `ordqz`, `hessenberg` and
`cossin`, matching SciPy within documented tolerance. (oracle: scipy/linalg/_decomp_schur.py, _decomp_qz.py)

#### Scenario: Schur form reconstructs the input
- GIVEN a square matrix `A`
- WHEN `schur(A)` returns `T, Z`
- THEN `Z @ T @ Z.conj().T` reconstructs `A` with `T` (quasi-)triangular, matching SciPy

### Requirement: Banded and structured solvers
`scipp::linalg` SHALL provide `solve_banded`, `solveh_banded`, `eig_banded`,
`eigh_tridiagonal` and `solve_triangular`, matching SciPy. (oracle: scipy/linalg/_basic.py, _decomp.py)

#### Scenario: Banded solve matches the dense solve
- GIVEN a banded system in diagonal-ordered form
- WHEN `solve_banded` is called
- THEN the solution matches the equivalent dense `solve` and SciPy

### Requirement: Matrix-equation solvers
`scipp::linalg` SHALL provide `solve_sylvester`, `solve_lyapunov`,
`solve_discrete_lyapunov`, `solve_continuous_are` and `solve_discrete_are`,
matching SciPy within documented tolerance. (oracle: scipy/linalg/_solvers.py)

#### Scenario: Sylvester / Lyapunov residual is near zero
- GIVEN the coefficient matrices of a Sylvester or Lyapunov equation
- WHEN the corresponding solver is called
- THEN the equation residual at the returned `X` is near zero, matching SciPy
