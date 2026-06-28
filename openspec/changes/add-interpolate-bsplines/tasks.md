# Tasks — B-splines

- [x] `BSpline` (t, c, k) representation with de Boor evaluation + extrapolation in `src/interpolate/bspline.cpp`
- [x] `make_interp_spline(x, y, k)` default not-a-knot knots + banded collocation solve via `numpp::linalg::solve`
- [x] `splev(x, tck)` de Boor evaluation wrapper
- [x] header declarations in `include/scypp/interpolate/interpolate.hpp`
- [x] extend oracle generator (scipy t/c/k, sampled evaluation, splev) for k=1,2,3; regenerate
- [x] `tests/test_interpolate_bspline.cpp` vs scipy golden + node-interpolation + validation
- [x] full suite green; `openspec validate add-interpolate-bsplines --strict`
- [x] trim BSpline/make_interp_spline/splev from `add-interpolate-splines` backlog
