# Design — interpolate (Phase 6)

## Context

`scypp::interpolate` ports the deterministic interpolators of
`scipy.interpolate`. SciPy's interpolators are objects: build from sample data,
then call at query points. ScyPP mirrors this with classes exposing
`operator()(const ndarray&)` (and a scalar overload). Tridiagonal spline systems
and the RBF linear solve go through `numpp::linalg::solve`.

## API shape

```cpp
namespace scypp::interpolate {
class Interp1d {
 public:
  Interp1d(const ndarray& x, const ndarray& y, std::string kind = "linear",
           std::optional<double> fill_value = std::nullopt);
  ndarray operator()(const ndarray& xq) const;
  double  operator()(double xq) const;
};
class CubicSpline { /* bc_type; operator()(xq, nu=0) */ };
class PchipInterpolator { /* operator()(xq) */ };
class Akima1DInterpolator { /* operator()(xq) */ };
class RegularGridInterpolator {        // points: per-axis grids; values: N-D
  RegularGridInterpolator(std::vector<ndarray> points, ndarray values, std::string method="linear");
  ndarray operator()(const ndarray& xi) const;   // xi shape (m, ndim)
};
class RBFInterpolator {                // y: (n, d) centers; d: (n,) values
  RBFInterpolator(const ndarray& y, const ndarray& vals, std::string kernel="thin_plate_spline",
                  double epsilon = 1.0, int degree = -1);
  ndarray operator()(const ndarray& x) const;     // x shape (m, d)
};
ndarray interpn(std::vector<ndarray> points, ndarray values, const ndarray& xi,
                std::string method = "linear");
}
```

## Algorithms

- **Interp1d** — sorted-`x` binary search per query; `linear` blends the bracket,
  `nearest` rounds to the closer sample, `previous`/`next` step. Out-of-bounds
  returns `fill_value` (default: error like SciPy's `bounds_error=True`, or NaN if a
  fill is given).
- **CubicSpline** — solve for the second derivatives `M` from the standard
  tridiagonal system; boundary rows implement `not-a-knot` (default), `natural`
  (`M₀=Mₙ=0`), or `clamped` (endpoint first-derivative). Evaluate the piecewise
  cubic (and its `nu`-th derivative) on the located interval. Matches SciPy, which
  also defaults to not-a-knot.
- **PchipInterpolator** — Fritsch–Carlson monotone slopes (harmonic-mean of secant
  slopes with the endpoint shape-preserving formula), then piecewise cubic Hermite.
- **Akima1DInterpolator** — Akima slopes from weighted secant differences (SciPy's
  default `"akima"` variant), then cubic Hermite.
- **RegularGridInterpolator** — locate the cell on each axis by binary search;
  `linear` does the `2ⁿ`-corner multilinear blend, `nearest` picks the closest
  corner. `interpn` is a thin wrapper.
- **RBFInterpolator** — assemble `A_ij = φ(‖yᵢ−yⱼ‖)` augmented with a degree-`d`
  polynomial block `P`; solve `[[A, P],[Pᵀ,0]] [w;c] = [vals;0]` via
  `numpp::linalg::solve`; evaluate `Σ wᵢ φ(‖x−yᵢ‖) + poly(x)`. Kernels:
  `thin_plate_spline` (`r²·log r`), `multiquadric`, `linear` (`r`), `cubic` (`r³`),
  `gaussian` (`exp(−(εr)²)`). Default `degree` follows SciPy per kernel.

## Oracle strategy

Each interpolator is built from the same samples in Python and C++, then evaluated
at shared query points; results are asserted `allclose` to SciPy (and, where
applicable, to the exact underlying function for smooth data). Spline derivatives
are checked against SciPy's `nu`-th derivative output. Tolerances are tight
(`~1e-9`) since these are deterministic linear/spline solves.

## Open questions

- SciPy's Akima has a modified-endpoint variant (`method="makima"`); this change
  implements the classic Akima (`method="akima"`, SciPy default) and tests against
  it. `makima` can be added later via a constructor flag.
