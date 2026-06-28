# Tasks — sparse extras (backlog, not yet scheduled)

> GitHub issue: [#6](https://github.com/CyberdyneCorp/SciPP/issues/6)

> Tracking artifact. Each item graduates into real implementation when picked up.

- [x] GPU CSR SpMV device kernel (CUDA/OpenCL/Metal) registered into NumPP's device vtable — delivered via `add-gpu-acceleration` (SpMV now delegates to `numpp::csr_spmv`, which auto-selects the device backend)
- [ ] `sparse.linalg`: `eigsh`, `eigs`, `svds`
- [ ] `splu`, `spilu`, `factorized`
- [ ] `DIA`, `LIL`, `BSR` formats + operations
- [x] `csgraph`: `breadth_first_order`, `depth_first_order`, `johnson`, `maximum_flow`, `maximum_bipartite_matching` — delivered via add-sparse-csgraph-traversal (strong/weak component modes still deferred)
- [ ] oracle tests for each
