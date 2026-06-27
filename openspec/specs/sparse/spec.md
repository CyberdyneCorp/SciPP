# sparse Specification

## Purpose
TBD - created by archiving change add-sparse. Update Purpose after archive.
## Requirements
### Requirement: Sparse matrix formats and conversion

`scypp::sparse` SHALL provide `CooMatrix`, `CsrMatrix` and `CscMatrix` with
construction from triplets and from dense, conversion between formats, `toarray`,
`nnz`, `transpose` and `diagonal`, matching SciPy. (oracle: scipy/sparse)

#### Scenario: Format round-trip
- GIVEN a dense matrix `A`
- WHEN it is converted to CSR/CSC/COO and back with `toarray`
- THEN the result equals `A`, `nnz` counts the nonzeros, and the canonical CSR
  matches SciPy's `csr_array(A)`

#### Scenario: Transpose and diagonal
- GIVEN a sparse matrix
- WHEN `transpose` and `diagonal` are computed
- THEN they match SciPy's

### Requirement: Sparse constructors and arithmetic

`scypp::sparse` SHALL provide `eye`, `identity` and `diags`, sparse+sparse
addition, and scalar multiplication, matching SciPy. (oracle: scipy/sparse/_construct.py)

#### Scenario: Constructors match SciPy
- WHEN `eye(n)` and `diags(diagonals, offsets)` are built and densified
- THEN they equal SciPy's

#### Scenario: Sparse addition
- GIVEN two sparse matrices of the same shape
- WHEN they are added
- THEN the dense result equals the dense sum and matches SciPy

### Requirement: Sparse matrix products with backend dispatch

`scypp::sparse` SHALL provide `spmv` (CSR · dense vector) and `spmm` (CSR · dense
matrix). `spmv` SHALL select an implementation via NumPP's capability registry —
a portable CPU kernel is always present and a device kernel is used when available
and the problem is large enough — with the chosen backend observable and the
device result equal to the CPU result within tolerance. (oracle: scipy/sparse)

#### Scenario: SpMV/SpMM match the dense product
- GIVEN a sparse matrix and a dense vector/matrix
- WHEN `spmv`/`spmm` are computed
- THEN the results equal the dense `A @ x` and match SciPy

#### Scenario: Backend dispatch and equivalence
- GIVEN an SpMV run once on the CPU path and once on the device-reference path
- WHEN the selected backend is queried
- THEN the two results are equal within tolerance and the backend is reported,
  with the CPU kernel always available as fallback

### Requirement: Sparse linear solvers

`scypp::sparse` SHALL provide `spsolve` (direct), `cg` and `gmres` (matrix-free
iterative), and `norm`, matching SciPy within documented tolerance. (oracle: scipy/sparse/linalg)

#### Scenario: Direct and iterative solves
- GIVEN a sparse system `A x = b`
- WHEN `spsolve(A, b)`, `cg(A, b)` (SPD `A`) and `gmres(A, b)` are computed
- THEN each returns `x` with `A @ x` ≈ `b`, matching SciPy

### Requirement: Compressed-sparse graph algorithms

`scypp::sparse::csgraph` SHALL provide `dijkstra`, `bellman_ford`,
`floyd_warshall`, `connected_components` and `minimum_spanning_tree` over a CSR
graph, matching SciPy. (oracle: scipy/sparse/csgraph)

#### Scenario: Shortest paths
- GIVEN a weighted graph as a CSR matrix
- WHEN `dijkstra` (and `bellman_ford`, `floyd_warshall`) compute shortest paths
- THEN the distance matrices match SciPy's

#### Scenario: Components and spanning tree
- GIVEN a graph
- WHEN `connected_components` and `minimum_spanning_tree` are computed
- THEN the component labeling matches SciPy (up to relabeling) and the MST total
  weight matches SciPy's

