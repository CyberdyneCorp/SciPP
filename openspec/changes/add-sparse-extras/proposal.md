# Sparse extras (deferred backlog)

## Why

Phase 9 (`add-sparse`, archived) delivered the sparse core and the SpMV
backend-dispatch architecture. This change tracks the deferred pieces — the actual
GPU device kernel, eigensolvers, the remaining formats, and sparse factorizations.
Not implemented yet.

## What changes

Adds (as target requirements) to the **sparse** capability:

- **GPU CSR SpMV device kernel**: a real CUDA/OpenCL/Metal CSR matrix-vector
  kernel registered into NumPP's weak-linked device vtable (requires a NumPP
  sparse backend), behind `SCYPP_WITH_*` flags with the CPU fallback.
- **Eigensolvers**: `sparse.linalg.eigsh`, `eigs`, `svds` (Lanczos/Arnoldi).
- **Direct factorizations**: `splu`, `spilu`, `factorized`.
- **Additional formats**: `DIA`, `LIL`, `BSR` with full operations.
- **More csgraph**: `johnson`, `shortest_path` dispatch, `breadth_first_order`,
  `depth_first_order`, `maximum_flow`, `maximum_bipartite_matching`,
  `strongly`/`weakly` connected component modes.

## Non-goals
- Implementing anything here; tracking only.
