# Tasks — interpolate splines & scattered data (backlog, not yet scheduled)

> Tracking artifact. Each item graduates into real implementation when picked up.

- [ ] `BSpline`, `make_interp_spline`, `splrep`/`splev`, `splint`/`sproot`
- [ ] `UnivariateSpline`, `InterpolatedUnivariateSpline`, `LSQUnivariateSpline`, `splprep`
- [x] `griddata` (2-D nearest/linear) — delivered via `add-interpolate-griddata`; cubic/Clough-Tocher and N-D `LinearNDInterpolator`/`NearestNDInterpolator` still pending
- [ ] `BarycentricInterpolator`, `KroghInterpolator`, makima, spline antiderivative/integral
- [ ] oracle tests for each
