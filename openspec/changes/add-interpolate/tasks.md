# Tasks — interpolate (Phase 6)

## 1. Module scaffold
- [ ] `include/scypp/interpolate/interpolate.hpp` (interpolator classes); `src/interpolate/*.cpp` in `src/CMakeLists.txt`; export from `scypp/scypp.hpp`
- [ ] Shared interval-search helper over sorted breakpoints

## 2. 1-D interpolation
- [ ] `Interp1d` kinds `"linear"`, `"nearest"`, `"previous"`, `"next"`; `fill_value` / bounds handling

## 3. Cubic splines
- [ ] `CubicSpline` tridiagonal solve for `not-a-knot` / `natural` / `clamped`
- [ ] Spline + derivative (`nu`) evaluation

## 4. Monotone / Akima
- [ ] `PchipInterpolator` (Fritsch–Carlson slopes + cubic Hermite)
- [ ] `Akima1DInterpolator` (Akima slopes + cubic Hermite)

## 5. N-D regular grid + RBF
- [ ] `RegularGridInterpolator` (`"linear"` multilinear, `"nearest"`) + `interpn`
- [ ] `RBFInterpolator` (kernels thin_plate_spline/multiquadric/linear/cubic/gaussian + polynomial tail via `numpp::linalg::solve`)

## 6. Oracle + validation
- [ ] Extend `tests/oracle/generate.py` with interpolate golden samples/queries; regenerate
- [ ] `tests/test_interpolate.cpp`: each interpolator vs SciPy at query points + node reproduction + spline derivative
- [ ] CPU build green; full suite green
- [ ] `openspec validate add-interpolate --strict` green
- [ ] Check off Phase 6 in `bootstrap-scypp-foundation/tasks.md`; update README
