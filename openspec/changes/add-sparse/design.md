# Design — sparse (Phase 9)

## Formats

Three classes over `numpp::ndarray`-backed coefficient arrays:

```cpp
namespace scypp::sparse {
struct CooMatrix { ndarray data, row, col; int64_t rows, cols; };   // triplets
class CsrMatrix {                                                    // workhorse
 public:
  CsrMatrix(ndarray data, ndarray indices, ndarray indptr, int64_t rows, int64_t cols);
  static CsrMatrix from_coo(const CooMatrix&);
  static CsrMatrix from_dense(const ndarray&);
  ndarray toarray() const;
  int64_t nnz() const;
  CscMatrix transpose() const;        // CSR^T is CSC of the same data
  ndarray diagonal() const;
  CsrMatrix add(const CsrMatrix&) const;
  CsrMatrix scaled(double) const;
  ndarray spmv(const ndarray& x) const;     // CSR · dense vector
  ndarray spmm(const ndarray& X) const;     // CSR · dense matrix
 private:
  ndarray data_, indices_, indptr_; int64_t rows_, cols_;
};
class CscMatrix { /* mirror, column-compressed */ };
}
```

`from_coo` sums duplicates and sorts indices within each row (SciPy canonical
form). `toarray` scatters into a dense `ndarray`. CSR is the canonical interchange
format; CSC is `transpose`.

## SpMV dispatch (GPU-ready)

`spmv` chooses a backend from `(nnz, available backends)` via NumPP's
`CapabilityRegistry`: above a size threshold and when a device CSR kernel is
available it would offload, otherwise the portable CPU kernel runs
(`y[i] = Σ_{k∈row i} data[k]·x[indices[k]]`). The chosen backend is recorded and
queryable (`last_backend()`), mirroring Phase-2 GEMM. The CPU kernel is always
present; a CPU-reference "device" path exercises the dispatch and proves
cross-backend equivalence without requiring hardware. The actual CUDA/OpenCL CSR
kernel is out of scope here (needs a NumPP sparse backend) and tracked separately.

## sparse.linalg

- `spsolve(A, b)` — convert `A` to dense and solve via `scypp::linalg::solve` (a
  true sparse LU is a follow-up; the result matches SciPy for the supported sizes).
- `cg(A, b)` — conjugate gradient, matrix-free through `spmv`, for SPD `A`.
- `gmres(A, b)` — restarted GMRES, matrix-free through `spmv`.
- `norm(A)` — Frobenius / 1 / inf norms of the nonzeros.

## sparse.csgraph

Graph algorithms over a CSR adjacency/weight matrix:

- `dijkstra` (binary-heap, non-negative weights), `bellman_ford` (negative edges,
  cycle detection), `floyd_warshall` (all-pairs).
- `connected_components` (union-find / BFS, directed weak/strong via SciPy's
  default), `minimum_spanning_tree` (Prim over the CSR, returned as a CSR tree).

Directed/undirected handling and `unweighted` follow SciPy's conventions.

## Oracle strategy

Compared `allclose` (or exactly, for index structure) to `scipy.sparse`: format
round-trips (`toarray`), `spmv`/`spmm` vs dense matmul and SciPy, `eye`/`diags`,
`spsolve`/`cg`/`gmres` residuals and solutions, and `csgraph` distances/components/
MST weights. The SpMV equivalence scenario compares the CPU and CPU-reference
"device" paths.

## Open questions

- `connected_components` directed `strong`/`weak` modes and MST tie-breaking
  follow SciPy; if ordering differs, tests compare the canonical (sorted) result.
