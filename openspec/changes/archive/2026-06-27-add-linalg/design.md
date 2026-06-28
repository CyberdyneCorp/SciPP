# Design — linalg (Phase 2)

## Context

`scipp::linalg` ports `scipy.linalg` on top of NumPP. NumPP already implements the
core dense decompositions (LAPACK-style `solve`/`inv`/`det`/`qr`/`svd`/`eig`/
`eigh`/`lstsq`/`pinv`/`norm`) and a GPU-routable `matmul`. SciPP therefore
**wraps and re-conventions** those, and **adds** the SciPy-only routines that
NumPP lacks. No new device code is written; acceleration is inherited through
NumPP's `matmul`/BLAS path.

## Delegate vs. implement

| SciPy routine | Strategy |
|---|---|
| `inv`, `det`, `solve`, `lstsq`, `pinv`, `norm` | delegate to `numpp::linalg`, adapt result shapes/args to SciPy |
| `qr`, `svd`, `svdvals`, `eig`, `eigvals`, `eigh`, `eigvalsh` | delegate to `numpp::linalg`, adapt conventions |
| `cholesky` | delegate to NumPP (lower) then transpose for SciPy's **upper default**; `lower=` flag |
| `pinvh` | implement via `eigh` (Hermitian pseudo-inverse) |
| `lu`, `lu_factor`, `lu_solve` | **implement** (partial-pivot Doolittle; SciPy-only P·L·U) |
| `ldl` | **implement** (Bunch–Kaufman-lite for symmetric indefinite; tests use the diagonal-pivot path) |
| `cho_factor`, `cho_solve` | **implement** (thin wrappers over `cholesky` + triangular solves) |
| `expm` | **implement** (Padé scaling-and-squaring, real arithmetic) |
| `polar` | **implement** via NumPP `svd`: `U = u·vh`, `P = vhᵀ·diag(s)·vh` |
| special matrices | **implement** by direct buffer fill |

## Key algorithms

- **LU (partial pivoting).** Doolittle with row pivoting on a row-major working
  copy; returns `p`/`l`/`u` for `lu`, and `(lu, piv)` for `lu_factor` matching
  SciPy's packed LAPACK `getrf` layout (`piv` are the row swaps). `lu_solve`
  applies the pivots then forward/back substitution. This is the one routine where
  exact P·L·U reconstruction is asserted against the oracle.
- **expm (Higham scaling-and-squaring).** Compute the matrix ∞-norm, choose a
  squaring power `s` so `‖A/2ˢ‖ ≤ θ₁₃`, evaluate the order-13 Padé numerator `U`
  and denominator `V` from precomputed coefficients using `numpp::matmul` for the
  matrix powers, solve `(V−U) X = (V+U)` with `numpp::linalg::solve`, then square
  `X` `s` times. All real; no eigendecomposition. Matches `scipy.linalg.expm`
  within ~1e-10.
- **polar.** `a = U P`; from `svd(a) = u·diag(s)·vh`, `U = u·vh` (nearest
  orthogonal) and `P = vhᵀ·diag(s)·vh` (SPD). Exact in real arithmetic; matches
  the SciPy oracle.
- **Special matrices.** Pure index fills (`toeplitz`, `circulant`, `hankel`,
  `tri`), block assembly (`block_diag`, `leslie`, `companion`), and recurrences
  (`pascal`, `hadamard` via Sylvester doubling, `hilbert`).

## GEMM / GPU path

`expm`, `lu_solve`, and any routine doing `A @ B` call `numpp::matmul`, which
dispatches to BLAS or a GPU backend when NumPP is built with one (size-threshold
gated) and otherwise the portable CPU kernel — results equal within tolerance and
the chosen backend is observable via `numpp::last_backend()`. No GPU device is
required to build or test; this change validates CPU correctness and documents the
accelerated path.

## Conventions & dtypes

- Results are `numpp::ndarray`. General `eig`/`eigvals` may be complex
  (delegated to NumPP); tests compare eigenvalues by sorted value and verify
  `A v = λ v` reconstruction rather than positional equality.
- `cholesky` defaults to the **upper** factor (SciPy), unlike NumPy's lower —
  a `lower` flag selects the other.
- Singular/non-conformable inputs raise `numpp::linalg_error` (surfaced from
  NumPP) or `scipp::value_error` for bad arguments, matching SciPy's `LinAlgError`
  / `ValueError` split.

## Oracle strategy

Extend `tests/oracle/generate.py` to emit golden matrices and SciPy results for
each routine. Decompositions are validated by **reconstruction** (`P@L@U ≈ A`,
`Q@R ≈ A`, `U@diag(s)@Vh ≈ A`) and by `allclose` to the SciPy factors where the
factorization is canonical; `expm`/`polar`/special matrices compare element-wise
to SciPy. Tolerances are per-routine (`rtol ≤ 1e-10` for well-conditioned inputs).

## Open questions

- `ldl` general Bunch–Kaufman pivoting is intricate; this change targets the
  symmetric quasi-definite diagonal-pivot case and defers 2×2 pivot blocks. Flagged
  in the spec; broaden in a follow-up if needed.
