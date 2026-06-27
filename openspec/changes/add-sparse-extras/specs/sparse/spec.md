# sparse Specification

## ADDED Requirements

### Requirement: GPU CSR matrix-vector kernel
`scypp::sparse::spmv` SHALL use a real CUDA/OpenCL/Metal CSR device kernel
(registered into NumPP's weak-linked device vtable, gated by `SCYPP_WITH_*`) when
the device is available and the problem is large enough, producing a result equal
to the CPU kernel within tolerance, with the CPU kernel always as fallback.
(oracle: scipy/sparse — value parity)

#### Scenario: Device SpMV equals the CPU result
- GIVEN a build with a sparse GPU backend and a usable device
- WHEN `spmv` runs on a large CSR matrix above the size threshold
- THEN the device result equals the CPU result within tolerance and `last_backend()`
  reports the device

### Requirement: Sparse eigensolvers and factorizations
`scypp::sparse::linalg` SHALL provide `eigsh`/`eigs`/`svds` (iterative
eigen/singular solvers) and `splu`/`spilu` (sparse LU), matching SciPy within
documented tolerance. (oracle: scipy/sparse/linalg)

#### Scenario: Largest eigenpairs match SciPy
- GIVEN a sparse symmetric matrix
- WHEN `eigsh(A, k)` computes the `k` largest eigenpairs
- THEN the eigenvalues are `allclose` to SciPy's and `A v ≈ λ v`

### Requirement: Additional formats and graph routines
`scypp::sparse` SHALL provide the `DIA`, `LIL` and `BSR` formats and the
additional `csgraph` routines (`breadth_first_order`, `depth_first_order`,
`johnson`, `maximum_flow`), matching SciPy. (oracle: scipy/sparse, scipy/sparse/csgraph)

#### Scenario: Additional formats round-trip
- GIVEN a dense matrix
- WHEN converted to DIA/LIL/BSR and back
- THEN the result equals the original and matches SciPy
