# linalg Specification

## ADDED Requirements

### Requirement: Basic linear algebra operations

`scypp::linalg` SHALL provide `inv`, `det`, `solve`, `lstsq`, `pinv`, `pinvh` and
`norm` over `numpp::ndarray`, matching SciPy within documented tolerance. Singular
systems SHALL raise an error mirroring SciPy's `LinAlgError`. (oracle: scipy/linalg/_basic.py)

#### Scenario: Solve and invert
- GIVEN a square nonsingular matrix `A` and right-hand side `b`
- WHEN `solve(A, b)` and `inv(A)` are called
- THEN `A @ solve(A,b)` is `allclose` to `b`, `A @ inv(A)` is `allclose` to the
  identity, and both match SciPy

#### Scenario: Least squares and pseudo-inverse
- GIVEN a tall matrix `A` and right-hand side `b`
- WHEN `lstsq(A, b)` and `pinv(A)` are called
- THEN the solution and pseudo-inverse are `allclose` to SciPy's, and `pinvh`
  matches SciPy for a Hermitian input

#### Scenario: Singular matrix
- GIVEN a singular square matrix
- WHEN `inv` or `solve` is called
- THEN an error is raised (mirroring SciPy `LinAlgError`)

### Requirement: Matrix decompositions

`scypp::linalg` SHALL provide `lu` / `lu_factor` / `lu_solve`, `qr`, `svd` /
`svdvals`, and `cholesky` / `cho_factor` / `cho_solve`, matching SciPy.
`cholesky` SHALL return the upper factor by default (SciPy convention) with a
`lower` option. (oracle: scipy/linalg/_decomp_lu.py, _decomp_qr.py, _decomp_svd.py,
_decomp_cholesky.py)

#### Scenario: LU reconstructs the input
- GIVEN a square matrix `A`
- WHEN `lu(A)` returns `P, L, U`
- THEN `P @ L @ U` is `allclose` to `A`, `L` is unit-lower-triangular, `U` is
  upper-triangular, and the result matches SciPy

#### Scenario: lu_factor / lu_solve round-trip
- GIVEN `A` and `b`
- WHEN `x = lu_solve(lu_factor(A), b)`
- THEN `A @ x` is `allclose` to `b`

#### Scenario: QR and SVD reconstruct the input
- GIVEN a matrix `A`
- WHEN `qr(A)` and `svd(A)` are computed
- THEN `Q @ R` and `U @ diag(s) @ Vh` are each `allclose` to `A`, with `s`
  non-negative and descending, matching SciPy

#### Scenario: Cholesky upper by default
- GIVEN a Hermitian positive-definite matrix `A`
- WHEN `cholesky(A)` is called with no `lower` flag
- THEN it returns an upper-triangular `R` with `R.conj().T @ R` ≈ `A` (SciPy
  upper convention), and `cho_solve(cho_factor(A), b)` solves `A x = b`

#### Scenario: Reject non-positive-definite Cholesky input
- GIVEN a matrix that is not positive definite
- WHEN `cholesky(A)` is called
- THEN an error is raised (mirroring SciPy `LinAlgError`)

### Requirement: Eigenvalue problems

`scypp::linalg` SHALL provide `eig`, `eigvals`, `eigh` and `eigvalsh`, matching
SciPy. `eigh`/`eigvalsh` SHALL return real ascending eigenvalues for Hermitian
input; `eig`/`eigvals` MAY return complex results. (oracle: scipy/linalg/_decomp.py)

#### Scenario: General eigenproblem reconstructs eigenpairs
- GIVEN a square matrix `A`
- WHEN `eig(A)` returns eigenvalues `w` and eigenvectors `V`
- THEN for each pair `A @ v` is `allclose` to `w * v`, and `eigvals(A)` matches
  SciPy's eigenvalues (compared as sorted sets)

#### Scenario: Hermitian eigenvalues are real and ascending
- GIVEN a Hermitian (real symmetric) matrix `A`
- WHEN `eigh(A)` / `eigvalsh(A)` are called
- THEN the eigenvalues are real, in ascending order, and `allclose` to SciPy

### Requirement: Matrix functions expm and polar

`scypp::linalg` SHALL provide `expm` (matrix exponential via Padé
scaling-and-squaring) and `polar` (polar decomposition via SVD), matching SciPy
within documented tolerance. (oracle: scipy/linalg/_matfuncs.py, _decomp_polar.py)

#### Scenario: Matrix exponential matches SciPy
- GIVEN a square matrix `A`
- WHEN `expm(A)` is computed
- THEN the result is `allclose` to `scipy.linalg.expm(A)`, and `expm` of the zero
  matrix is the identity

#### Scenario: Polar decomposition factors the input
- GIVEN a matrix `A`
- WHEN `polar(A)` returns `U, P`
- THEN `U @ P` is `allclose` to `A`, `U` has orthonormal columns, `P` is symmetric
  positive-semidefinite, and both match SciPy

### Requirement: Special-matrix constructors

`scypp::linalg` SHALL provide `toeplitz`, `circulant`, `hankel`, `block_diag`,
`companion`, `hilbert`, `hadamard`, `pascal`, `leslie`, `kron` and `tri`, matching
SciPy element-wise. (oracle: scipy/linalg/_special_matrices.py)

#### Scenario: Constructors match SciPy
- GIVEN the defining inputs for each constructor
- WHEN the ScyPP constructor is called
- THEN the resulting matrix is `allclose` (or exactly equal for integer fills) to
  SciPy's

#### Scenario: Structural properties hold
- GIVEN a `circulant(c)` and a `hadamard(n)` (n a power of two)
- WHEN they are constructed
- THEN `circulant` is constant along diagonals and `hadamard @ hadamard.T` is
  `n * I`

### Requirement: Accelerated GEMM-backed routines

Routines whose hot path is dense matrix multiply (including `expm`) SHALL use
NumPP's `matmul`, so that a BLAS- or GPU-enabled NumPP build accelerates them while
producing CPU-equivalent results within tolerance, with the selected backend
observable via NumPP. (oracle: scipy/linalg)

#### Scenario: Accelerated result equals the CPU result
- GIVEN the same input on a build where NumPP can use a BLAS/GPU backend
- WHEN a GEMM-backed routine runs once on the CPU path and once on the accelerated
  path
- THEN the two results are `allclose` within the documented tolerance
