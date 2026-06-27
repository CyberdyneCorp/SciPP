# Tasks — linalg (Phase 2)

## 1. Module scaffold
- [x] `include/scypp/linalg/linalg.hpp` (declarations) + result structs (`LUResult`, `LUFactor`, `QRResult`, `SVDResult`, `EigResult`, `EighResult`, `CholFactor`, `PolarResult`, `LstsqResult`)
- [x] `src/linalg/*.cpp` added to `src/CMakeLists.txt`; exported from `scypp/scypp.hpp`
- [x] Shared helpers: row-major float64 working-copy accessor, triangular solves

## 2. Basic operations (delegate to NumPP, SciPy conventions)
- [x] `inv`, `det`, `solve`, `lstsq`, `pinv`, `norm`
- [x] `pinvh` via `eigh`
- [x] Singular input raises (mirrors SciPy `LinAlgError`)

## 3. Decompositions
- [x] `lu` (partial-pivot Doolittle → P, L, U) and `lu_factor`/`lu_solve` (packed LU + pivots)
- [x] `qr`, `svd`, `svdvals` (delegate to NumPP)
- [x] `cholesky` (upper default + `lower` flag), `cho_factor`, `cho_solve`

## 4. Eigenvalue problems
- [x] `eig`, `eigvals`, `eigh`, `eigvalsh` (delegate to NumPP, adapt conventions)

## 5. Matrix functions
- [x] `expm` (order-13 Padé scaling-and-squaring via `numpp::matmul` + `solve`)
- [x] `polar` (SVD-based: `U = u·vh`, `P = vhᵀ·diag(s)·vh`)

## 6. Special matrices
- [x] `toeplitz`, `circulant`, `hankel`, `tri`
- [x] `block_diag`, `companion`, `leslie`, `kron`
- [x] `hilbert`, `hadamard` (Sylvester doubling), `pascal`

## 7. Oracle + validation
- [x] Extend `tests/oracle/generate.py` with linalg golden matrices/results; regenerate frozen data
- [x] `tests/test_linalg.cpp`: reconstruction checks (P·L·U, Q·R, U·S·Vh), `expm`/`polar`/special-matrix `allclose`, eigen reconstruction, singular-input error
- [x] CPU build green; full suite green against frozen oracle data
- [x] `openspec validate add-linalg --strict` green
- [x] Check off Phase 2 in `bootstrap-scypp-foundation/tasks.md`; update README status
