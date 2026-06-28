# Wire SciPP hot kernels to NumPP device acceleration

## Why

SciPP's architecture always intended GPU acceleration for SciPy's CPU-bound hot
kernels, routed through NumPP's tiered dispatch. Phases 9–11 shipped the dispatch
*shape* (CapabilityRegistry probe + `last_backend()`) but ran a hand-rolled CPU
loop in every case, because NumPP did not yet expose the device kernels. **NumPP
1.5.0 now ships them** — `csr_spmv`, `cdist_euclidean`, and `correlate1d` (issues
#99, #106) — each auto-selecting a device backend above a size threshold, falling
back to the portable CPU path, and recording the choice via `last_backend()`.

This change replaces SciPP's placeholder CPU loops with calls to those kernels, so
sparse SpMV, pairwise euclidean distance, and separable filtering gain real
CUDA/OpenCL/Metal acceleration through the same weak-linked vtable — with the CPU
path unchanged as the guaranteed fallback.

## What changes

- **sparse**: `spmv`/`spmm` (CSR) delegate to `numpp::csr_spmv`.
- **spatial**: `cdist`/`pdist` for `euclidean`/`sqeuclidean` delegate to
  `numpp::cdist_euclidean`.
- **ndimage**: `correlate1d` and the separable filters built on it
  (gaussian/uniform/derivative passes) delegate to `numpp::correlate1d`, mapping
  SciPP boundary-mode strings to `numpp::FilterMode`.
- Each path reports the chosen backend through `last_backend()` and adds a
  cross-backend equivalence test (CPU result equals the accelerated result, and
  the existing SciPy-oracle results still hold).
- Pins the NumPP dependency floor to **1.5.0**.

## Impact

- Affected specs: **modifies** the `sparse`, `spatial`, and `ndimage` capabilities
  (device-acceleration requirement each).
- Affected code: `src/sparse/ops.cpp`, `src/spatial/distance.cpp`,
  `src/ndimage/filters.cpp`, `CMakeLists.txt` (version floor), tests.
- Trims the "GPU CSR SpMV device kernel" item in `add-sparse-extras` and the
  deferred GPU-kernel notes from the Phase 10/11 baselines.

## Non-goals

- Re-implementing device kernels in SciPP — they live in NumPP; SciPP only offloads.
- Non-euclidean `cdist` metrics and non-separable `ndimage` filters (stay CPU).
- A GPU CI runner: local builds exercise the CPU fallback; true device equivalence
  is validated where a NumPP GPU variant is present.

## Build-cleanup decisions (finalize step)

- Dropped `-DNUMPP_WARNINGS_AS_ERRORS=OFF` from `scripts/bootstrap_numpp.sh`: NumPP
  1.4.0/1.5.0 default this OFF and fixed the datetime warning, so NumPP bootstraps
  and installs cleanly without the override.
- Kept the cherry-picked `numpp/<module>/<header>.hpp` includes instead of the
  `numpp/numpp.hpp` umbrella. They are correct and span the whole `src/` tree;
  switching wholesale is churn with no functional gain and a regression risk. The
  umbrella umbrella-install fix (issue #107) is available if a future change wants it.
- Kept `NO_CMAKE_PACKAGE_REGISTRY` / `NO_CMAKE_SYSTEM_PACKAGE_REGISTRY` and the
  pinned `NumPP_DIR` in `CMakeLists.txt`: the local bootstrap-install model
  (`.deps/numpp`) relies on resolving exactly that prefix and not a stray system
  registry entry. Harmless and load-bearing for reproducible local builds.
