# backend-acceleration Specification

## Purpose
TBD - created by archiving change bootstrap-scipp-foundation. Update Purpose after archive.
## Requirements
### Requirement: Reuse NumPP's device dispatch substrate

SciPP SHALL NOT implement a second device-management stack. It SHALL reuse NumPP's
`CapabilityRegistry` (compiled-backend and present-device probing),
`last_backend()` introspection, the `NUMPP_GPU_TARGET` selection override, and the
bounded device buffer reuse pool. SciPP GPU kernels SHALL be registered into the
same weak-linked vtable shape NumPP uses.

#### Scenario: SciPP queries the shared registry
- WHEN SciPP needs to decide whether a GPU path is available
- THEN it queries NumPP's `CapabilityRegistry` rather than probing devices itself

#### Scenario: No parallel device runtime
- WHEN SciPP is built with a GPU flag enabled
- THEN device memory and kernel launches go through NumPP's device layer and pool,
  not a SciPP-owned allocator

### Requirement: SciPy-specific accelerated kernels

SciPP SHALL provide optional GPU implementations for the hot kernels SciPy leaves
on the CPU — at minimum FFT, GEMM-backed dense linalg steps, sparse CSR
matrix-vector product (SpMV), separable convolution for `ndimage`, and pairwise
distance matrices for `spatial.distance` — each gated by its `SCIPP_WITH_<BACKEND>`
flag, built as a separate weak-linked translation unit, for **CUDA, OpenCL and
Metal**. A portable CPU implementation SHALL always exist for every such kernel.

#### Scenario: Accelerated kernel present and used above threshold
- GIVEN a build with a GPU backend compiled in and a usable device
- WHEN an eligible accelerated kernel runs on a problem above its size threshold
- THEN the device path is used and `last_backend()` reports that GPU backend

#### Scenario: CPU fallback always available
- GIVEN any accelerated kernel
- WHEN no GPU backend is compiled in, or no device is present, or the problem is
  below the size threshold, or the dtype is unsupported on device
- THEN the portable CPU implementation is used and the call still succeeds

### Requirement: Size-threshold and dtype-gated dispatch

Each accelerable SciPP kernel SHALL choose its implementation from `(operation,
dtype, problem size, available backends)`. Below a per-operation size threshold the
CPU kernel SHALL be used to avoid offload overhead; an accelerated backend SHALL be
used only when available and the problem is large enough and the dtype is supported.

#### Scenario: Small problem stays on CPU
- GIVEN a build with a GPU backend available
- WHEN an accelerable kernel runs below its configured size threshold
- THEN the CPU kernel is used and `last_backend()` reports CPU

#### Scenario: Unsupported dtype declines to CPU
- GIVEN a GPU kernel that supports only float32/float64
- WHEN it is invoked on an integer or unsupported-complex dtype
- THEN it declines to the CPU kernel

### Requirement: Backend selection override and Apple preference

SciPP SHALL honor NumPP's `NUMPP_GPU_TARGET` override (`cpu|cuda|opencl|metal|
auto`). `auto` SHALL prefer Metal on Apple platforms and otherwise try CUDA, then
OpenCL, then CPU. Requesting a backend not compiled in SHALL produce a clear error;
`auto` SHALL degrade silently to CPU.

#### Scenario: Force CPU
- GIVEN a build with a GPU backend available
- WHEN `NUMPP_GPU_TARGET=cpu` is set
- THEN all SciPP kernels use the CPU path

#### Scenario: Auto prefers Metal on Apple
- GIVEN an Apple build with the Metal backend compiled in and a usable device
- WHEN `NUMPP_GPU_TARGET=auto` is set
- THEN eligible kernels select the Metal backend

#### Scenario: Explicit unavailable backend errors
- GIVEN a build with `SCIPP_WITH_CUDA=OFF`
- WHEN `NUMPP_GPU_TARGET=cuda` is set and an accelerable kernel is requested on CUDA
- THEN a clear error is reported

### Requirement: Cross-backend result equivalence

Every accelerable SciPP kernel SHALL produce, on any accelerated backend, a result
equal to the portable CPU kernel's result within a documented tolerance, so that
enabling a backend never changes observable numerical behavior beyond rounding.

#### Scenario: Device and CPU agree within tolerance
- GIVEN the same input
- WHEN an accelerable kernel runs once forced to CPU and once forced to a GPU backend
- THEN the two results are equal within the kernel's documented tolerance
  (`allclose`)

#### Scenario: Backend choice is observable in tests
- GIVEN a kernel run above its threshold with a GPU backend forced
- WHEN the selected backend is queried via `last_backend()`
- THEN it reports the GPU backend, allowing tests to assert the path taken, not
  only the numeric result

