# Tasks — linalg (Phase 2)

## 1. Module scaffold
- [ ] `include/scypp/linalg/linalg.hpp` (declarations) + result structs (`LUResult`, `LUFactor`, `QRResult`, `SVDResult`, `EigResult`, `EighResult`, `LDLResult`, `CholFactor`, `PolarResult`)
- [ ] `src/linalg/*.cpp` added to `src/CMakeLists.txt`; exported from `scypp/scypp.hpp`
- [ ] Shared helpers: row-major float64 working-copy accessor, triangular solves

## 2. Basic operations (delegate to NumPP, SciPy conventions)
- [ ] `inv`, `det`, `solve`, `lstsq`, `pinv`, `norm`
- [ ] `pinvh` via `eigh`
- [ ] Singular input raises (mirrors SciPy `LinAlgError`)

## 3. Decompositions
- [ ] `lu` (partial-pivot Doolittle → P, L, U) and `lu_factor`/`lu_solve` (packed LU + pivots)
- [ ] `qr`, `svd`, `svdvals` (delegate to NumPP)
- [ ] `cholesky` (upper default + `lower` flag), `cho_factor`, `cho_solve`
- [ ] `ldl` (symmetric, diagonal-pivot path)

## 4. Eigenvalue problems
- [ ] `eig`, `eigvals`, `eigh`, `eigvalsh` (delegate to NumPP, adapt conventions)

## 5. Matrix functions
- [ ] `expm` (order-13 Padé scaling-and-squaring via `numpp::matmul` + `solve`)
- [ ] `polar` (SVD-based: `U = u·vh`, `P = vhᵀ·diag(s)·vh`)

## 6. Special matrices
- [ ] `toeplitz`, `circulant`, `hankel`, `tri`
- [ ] `block_diag`, `companion`, `leslie`, `kron`
- [ ] `hilbert`, `hadamard` (Sylvester doubling), `pascal`

## 7. Oracle + validation
- [ ] Extend `tests/oracle/generate.py` with linalg golden matrices/results; regenerate frozen data
- [ ] `tests/test_linalg.cpp`: reconstruction checks (P·L·U, Q·R, U·S·Vh), `expm`/`polar`/special-matrix `allclose`, eigen reconstruction, singular-input error
- [ ] CPU build green; full suite green against frozen oracle data
- [ ] `openspec validate add-linalg --strict` green
- [ ] Check off Phase 2 in `bootstrap-scypp-foundation/tasks.md`; update README status
