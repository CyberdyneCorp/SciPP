# Tasks — interpolate splines & scattered data (backlog, not yet scheduled)

> GitHub issue: [#7](https://github.com/CyberdyneCorp/SciPP/issues/7)

> Tracking artifact. Each item graduates into real implementation when picked up.

- [x] `BSpline`, `make_interp_spline`, `splev` — delivered via `add-interpolate-bsplines`; `splrep` (s>0 knot selection), `splint`/`sproot` still pending
- [ ] `UnivariateSpline`, `InterpolatedUnivariateSpline`, `LSQUnivariateSpline`, `splprep`
- [x] `griddata` (2-D nearest/linear) — delivered via `add-interpolate-griddata`; cubic/Clough-Tocher and N-D `LinearNDInterpolator`/`NearestNDInterpolator` still pending
- [ ] `BarycentricInterpolator`, `KroghInterpolator`, makima, spline antiderivative/integral
- [ ] oracle tests for each
