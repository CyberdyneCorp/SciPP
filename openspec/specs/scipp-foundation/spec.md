# scipp-foundation Specification

## Purpose
TBD - created by archiving change bootstrap-scipp-foundation. Update Purpose after archive.
## Requirements
### Requirement: Built on NumPP as the array engine

SciPP SHALL use NumPP as its array engine. Public SciPP APIs SHALL accept and
return `numpp::ndarray`; SciPP SHALL NOT define its own array, dtype, or
broadcasting type. Element-wise steps inside SciPP algorithms SHALL use NumPP
ufuncs/linalg/fft/random rather than re-implementing them. (oracle: scipy depends
on numpy)

#### Scenario: SciPP routine consumes and produces NumPP arrays
- GIVEN a `numpp::ndarray` input `A`
- WHEN a SciPP routine (e.g. a decomposition or transform) is called on `A`
- THEN it returns `numpp::ndarray` result(s) without converting to any SciPP-owned
  array type

#### Scenario: No duplicate array engine
- WHEN the SciPP public headers are inspected
- THEN there is no SciPP-defined N-D array, dtype enumeration, or broadcasting
  implementation; these are taken from NumPP

### Requirement: Portable CPU build with no extra dependencies

SciPP SHALL build, link, and pass its CPU test subset on any C++20 toolchain —
including iOS and Android cross-compilation toolchains — with all optional backend
feature flags OFF, requiring only NumPP and the C++ standard library.

#### Scenario: No-accel build is fully functional
- GIVEN a build with `SCIPP_WITH_BLAS=OFF`, `SCIPP_WITH_LAPACK=OFF` and all
  `SCIPP_WITH_<GPU>=OFF`
- WHEN the library is built and the CPU test subset is run
- THEN the build succeeds and the CPU tests pass

#### Scenario: Mobile toolchain build
- GIVEN the iOS and Android NDK cross-compilation toolchains
- WHEN SciPP is built with all backend flags OFF
- THEN compilation and linking succeed

### Requirement: CMake C++20 project layout and feature flags

SciPP SHALL be a CMake (≥ 3.25) C++20 project that surfaces backend feature flags
in a generated `config.hpp`: `SCIPP_WITH_BLAS`, `SCIPP_WITH_LAPACK`,
`SCIPP_WITH_CUDA`, `SCIPP_WITH_OPENCL`, `SCIPP_WITH_METAL`, all defaulting OFF.
Enabling a SciPP GPU flag SHALL enable the corresponding NumPP backend flag so the
two layers share one device runtime.

#### Scenario: Flags default off
- WHEN the project is configured with no backend options specified
- THEN all `SCIPP_WITH_*` flags are OFF in the generated config

#### Scenario: GPU flag propagates to NumPP
- WHEN SciPP is configured with `SCIPP_WITH_CUDA=ON`
- THEN NumPP is configured with its CUDA backend enabled and both layers resolve
  the same device runtime

### Requirement: Developer task runner

SciPP SHALL provide a `justfile` task runner as the canonical developer entrypoint,
mirroring the NumPP and SymPP sibling ports, wrapping the bootstrap, configure,
build, test, lint/spec, and clean workflows over the underlying CMake project.

#### Scenario: Common tasks are one command
- GIVEN a checkout with `just` installed
- WHEN a developer runs `just bootstrap` then `just test`
- THEN the pinned NumPP is installed under `.deps/` and the SciPy-oracle test
  suite builds and runs green

#### Scenario: Spec validation is a recipe
- WHEN a developer runs `just spec`
- THEN `openspec validate --all --strict` runs over every spec and change

### Requirement: Packaging mirrors the sibling ports

SciPP SHALL ship Conan (`conanfile.py`) and vcpkg (`vcpkg.json`) packaging that
declares NumPP as a dependency, mirroring the NumPP and SymPP packaging layout, so
a consumer can depend on SciPP and transitively obtain NumPP.

#### Scenario: Consumer obtains SciPP and NumPP together
- GIVEN a downstream project that depends on the SciPP package
- WHEN it resolves dependencies via Conan or vcpkg
- THEN NumPP is pulled in transitively at the pinned version

### Requirement: Ported exception and status model

SciPP SHALL provide an error model aligned with NumPP's: a `scipp::error` base
compatible with `numpp::error`, with typed categories mirroring SciPy
(`value_error`, `linalg_error`, `not_implemented_error`). Where SciPy returns a
status object instead of raising (e.g. `OptimizeResult.success`, integration error
codes), SciPP SHALL port that as a returned status struct rather than an exception.

#### Scenario: Single catch site spans both layers
- GIVEN code that calls both NumPP and SciPP routines
- WHEN either layer raises
- THEN a single `catch (const numpp::error&)` (or compatible base) handles both

#### Scenario: Non-convergent factorization raises linalg_error
- GIVEN an input for which a SciPP linear-algebra routine cannot converge or is
  singular
- WHEN the routine is called
- THEN it raises `scipp::linalg_error`

#### Scenario: Optimizer reports failure via status, not exception
- GIVEN an optimization that does not converge within its iteration budget
- WHEN the routine returns
- THEN it returns a result whose `success` flag is false and whose status message
  describes the failure, matching SciPy's behavior, without throwing

