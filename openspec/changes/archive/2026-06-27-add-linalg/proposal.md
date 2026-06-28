# Add linalg (Phase 2)

## Why

Phase 2 of the SciPP roadmap (see `bootstrap-scipp-foundation`). `scipy.linalg`
is the most widely depended-upon numerical subpackage and the first one that is
acceleration-sensitive: its hot path is dense matrix multiply (GEMM), which SciPP
routes through NumPP's BLAS/GPU backends. `optimize`, `integrate`, `signal`,
`stats` and `sparse` all build on it.

`scipy.linalg` overlaps `numpy.linalg` but adds routines (LU with explicit P/L/U,
LDL, matrix functions, special-matrix constructors) and uses different
conventions (e.g. `cholesky` returns the **upper** factor by default). SciPP
**delegates the standard decompositions to NumPP** (`solve`/`inv`/`det`/`qr`/
`svd`/`eig`/`eigh`/`lstsq`/`pinv`/`norm`), adapting return shapes to SciPy, and
**adds the SciPy-only routines** on top.

## What changes

Adds the **linalg** capability — `scipp::linalg`, ported against the SciPy oracle,
over `numpp::ndarray`:

- **Basic**: `inv`, `det`, `solve`, `lstsq`, `pinv`, `pinvh`, `norm`
  (matrix/vector orders).
- **Decompositions**: `lu` / `lu_factor` / `lu_solve` (the SciPy-only explicit
  P·L·U), `qr`, `svd` / `svdvals`, `cholesky` (SciPy upper-default) / `cho_factor`
  / `cho_solve`.
- **Eigenvalue problems**: `eig`, `eigvals`, `eigh`, `eigvalsh` (incl. the
  `eigvals_only` / subset-free standard forms).
- **Matrix functions**: `expm` (Padé scaling-and-squaring) and `polar`
  (SVD-based) — the two that need no Schur form and stay in real arithmetic.
- **Special matrices**: `toeplitz`, `circulant`, `hankel`, `block_diag`,
  `companion`, `hilbert`, `hadamard`, `pascal`, `leslie`, `kron`, `tri`.
- **Accelerated dispatch**: GEMM-backed routines (and `expm`'s repeated products)
  go through NumPP's `matmul`, so a BLAS/GPU-enabled build accelerates them with
  CPU-equivalent results.

## Impact

- Affected specs: **adds** the `linalg` capability.
- Affected code: new `include/scipp/linalg/`, `src/linalg/`, `tests/test_linalg.cpp`,
  extended oracle generator. Reuses the Phase 1 foundation unchanged.
- Roadmap: checks off Phase 2 in `bootstrap-scipp-foundation/tasks.md`.

## Non-goals (deferred to a later linalg change)

- **`ldl`**: SciPy's Bunch–Kaufman LDLᵀ with 2×2 pivot blocks is hard to match
  bit-for-bit against the oracle; deferred to its own change.
- **Matrix functions beyond expm/polar**: `sqrtm`, `logm`, `funm`, `cosm`,
  `sinm`, `fractional_matrix_power` — they need a robust complex Schur form to
  match SciPy on defective matrices.
- **Schur-family**: `schur`, `rsf2csf`, `qz`, `ordqz`, `cossin`, `hessenberg`.
- **Banded/structured solvers**: `solve_banded`, `solveh_banded`, `eig_banded`,
  `eigh_tridiagonal`, `solve_triangular` fast paths.
- **Matrix-equation solvers**: `solve_sylvester`, `solve_lyapunov`,
  `solve_*_riccati` (need Schur/Bartels–Stewart).
- **Low-level BLAS/LAPACK wrappers** (`scipy.linalg.blas`/`lapack` namespaces).
- No GPU kernels authored here beyond what NumPP's `matmul` already provides.
