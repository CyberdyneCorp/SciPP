# sparse Specification

## ADDED Requirements

### Requirement: Device-accelerated CSR sparse matrix-vector product
`scipp::sparse` SHALL compute CSR `spmv` and `spmm` through NumPP's `csr_spmv`
device kernel, auto-selecting a GPU backend above the size threshold, always
falling back to the portable CPU path, with the selection observable via
`last_backend()`. (oracle: scipy.sparse CSR matvec)

#### Scenario: SpMV is backend-independent and matches SciPy
- GIVEN a CSR matrix and a dense vector
- WHEN `spmv` runs (CPU, and a device backend when one is present)
- THEN every backend's result is `allclose` to the other and to `scipy.sparse`
- AND `last_backend()` reports the backend actually used
