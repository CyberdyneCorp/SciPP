# Advanced linalg (deferred backlog)

## Why

Phase 2 (`add-linalg`, archived) delivered the dense linear-algebra core but
deferred several routines. This change is the **tracked backlog** for them so the
work flows through OpenSpec when scheduled. It is **not implemented yet**.

## What changes

Adds (as target requirements) to the **linalg** capability:

- **LDL decomposition**: `ldl` (Bunch–Kaufman with 2×2 pivot blocks).
- **Matrix functions beyond expm/polar**: `sqrtm`, `logm`, `funm`, `cosm`,
  `sinm`, `fractional_matrix_power`.
- **Schur family**: `schur`, `rsf2csf`, `qz`, `ordqz`, `cossin`, `hessenberg`.
- **Banded/structured solvers**: `solve_banded`, `solveh_banded`, `eig_banded`,
  `eigh_tridiagonal`, `eigvalsh_tridiagonal`, `solve_triangular`.
- **Matrix-equation solvers**: `solve_sylvester`, `solve_lyapunov`,
  `solve_discrete_lyapunov`, `solve_continuous_are`, `solve_discrete_are`.
- **Low-level access**: the `scipy.linalg.blas` / `scipy.linalg.lapack` wrappers.

## Non-goals

- Implementing anything here; each unchecked task graduates into real work when
  picked up.
