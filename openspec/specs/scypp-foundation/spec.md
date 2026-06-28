# scypp-foundation Specification

## Purpose
TBD - created by archiving change bootstrap-scypp-foundation. Update Purpose after archive.
## Requirements
### Requirement: Built on NumPP as the array engine

ScyPP SHALL use NumPP as its array engine. Public ScyPP APIs SHALL accept and
return `numpp::ndarray`; ScyPP SHALL NOT define its own array, dtype, or
broadcasting type. Element-wise steps inside ScyPP algorithms SHALL use NumPP
ufuncs/linalg/fft/random rather than re-implementing them. (oracle: scipy depends
on numpy)

#### Scenario: ScyPP routine consumes and produces NumPP arrays
- GIVEN a `numpp::ndarray` input `A`
- WHEN a ScyPP routine (e.g. a decomposition or transform) is called on `A`
- THEN it returns `numpp::ndarray` result(s) without converting to any ScyPP-owned
  array type

#### Scenario: No duplicate array engine
- WHEN the ScyPP public headers are inspected
- THEN there is no ScyPP-defined N-D array, dtype enumeration, or broadcasting
  implementation; these are taken from NumPP

### Requirement: Portable CPU build with no extra dependencies

ScyPP SHALL build, link, and pass its CPU test subset on any C++20 toolchain —
including iOS and Android cross-compilation toolchains — with all optional backend
feature flags OFF, requiring only NumPP and the C++ standard library.

#### Scenario: No-accel build is fully functional
- GIVEN a build with `SCYPP_WITH_BLAS=OFF`, `SCYPP_WITH_LAPACK=OFF` and all
  `SCYPP_WITH_<GPU>=OFF`
- WHEN the library is built and the CPU test subset is run
- THEN the build succeeds and the CPU tests pass

#### Scenario: Mobile toolchain build
- GIVEN the iOS and Android NDK cross-compilation toolchains
- WHEN ScyPP is built with all backend flags OFF
- THEN compilation and linking succeed

### Requirement: CMake C++20 project layout and feature flags

ScyPP SHALL be a CMake (≥ 3.25) C++20 project that surfaces backend feature flags
in a generated `config.hpp`: `SCYPP_WITH_BLAS`, `SCYPP_WITH_LAPACK`,
`SCYPP_WITH_CUDA`, `SCYPP_WITH_OPENCL`, `SCYPP_WITH_METAL`, all defaulting OFF.
Enabling a ScyPP GPU flag SHALL enable the corresponding NumPP backend flag so the
two layers share one device runtime.

#### Scenario: Flags default off
- WHEN the project is configured with no backend options specified
- THEN all `SCYPP_WITH_*` flags are OFF in the generated config

#### Scenario: GPU flag propagates to NumPP
- WHEN ScyPP is configured with `SCYPP_WITH_CUDA=ON`
- THEN NumPP is configured with its CUDA backend enabled and both layers resolve
  the same device runtime

### Requirement: Packaging mirrors the sibling ports

ScyPP SHALL ship Conan (`conanfile.py`) and vcpkg (`vcpkg.json`) packaging that
declares NumPP as a dependency, mirroring the NumPP and SymPP packaging layout, so
a consumer can depend on ScyPP and transitively obtain NumPP.

#### Scenario: Consumer obtains ScyPP and NumPP together
- GIVEN a downstream project that depends on the ScyPP package
- WHEN it resolves dependencies via Conan or vcpkg
- THEN NumPP is pulled in transitively at the pinned version

### Requirement: Ported exception and status model

ScyPP SHALL provide an error model aligned with NumPP's: a `scypp::error` base
compatible with `numpp::error`, with typed categories mirroring SciPy
(`value_error`, `linalg_error`, `not_implemented_error`). Where SciPy returns a
status object instead of raising (e.g. `OptimizeResult.success`, integration error
codes), ScyPP SHALL port that as a returned status struct rather than an exception.

#### Scenario: Single catch site spans both layers
- GIVEN code that calls both NumPP and ScyPP routines
- WHEN either layer raises
- THEN a single `catch (const numpp::error&)` (or compatible base) handles both

#### Scenario: Non-convergent factorization raises linalg_error
- GIVEN an input for which a ScyPP linear-algebra routine cannot converge or is
  singular
- WHEN the routine is called
- THEN it raises `scypp::linalg_error`

#### Scenario: Optimizer reports failure via status, not exception
- GIVEN an optimization that does not converge within its iteration budget
- WHEN the routine returns
- THEN it returns a result whose `success` flag is false and whose status message
  describes the failure, matching SciPy's behavior, without throwing

