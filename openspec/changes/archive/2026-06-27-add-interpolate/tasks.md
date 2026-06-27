# Tasks — interpolate (Phase 6)

## 1. Module scaffold
- [x] `include/scypp/interpolate/interpolate.hpp` (interpolator classes); `src/interpolate/*.cpp` in `src/CMakeLists.txt`; export from `scypp/scypp.hpp`
- [x] Shared interval-search helper over sorted breakpoints

## 2. 1-D interpolation
- [x] `Interp1d` kinds `"linear"`, `"nearest"`, `"previous"`, `"next"`; `fill_value` / bounds handling

## 3. Cubic splines
- [x] `CubicSpline` tridiagonal solve for `not-a-knot` / `natural` / `clamped`
- [x] Spline + derivative (`nu`) evaluation

## 4. Monotone / Akima
- [x] `PchipInterpolator` (Fritsch–Carlson slopes + cubic Hermite)
- [x] `Akima1DInterpolator` (Akima slopes + cubic Hermite)

## 5. N-D regular grid + RBF
- [x] `RegularGridInterpolator` (`"linear"` multilinear, `"nearest"`) + `interpn`
- [x] `RBFInterpolator` (kernels thin_plate_spline/multiquadric/linear/cubic/gaussian + polynomial tail via `numpp::linalg::solve`)

## 6. Oracle + validation
- [x] Extend `tests/oracle/generate.py` with interpolate golden samples/queries; regenerate
- [x] `tests/test_interpolate.cpp`: each interpolator vs SciPy at query points + node reproduction + spline derivative
- [x] CPU build green; full suite green
- [x] `openspec validate add-interpolate --strict` green
- [x] Check off Phase 6 in `bootstrap-scypp-foundation/tasks.md`; update README
