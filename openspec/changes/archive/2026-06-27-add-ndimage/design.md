# Design — ndimage (Phase 11)

Operates on 2-D `numpp::ndarray` images (the common case; the separable filters and
the EDT generalize to N-D and are written dimension-agnostically where practical).

## Boundary modes

A single `boundary_index(i, n, mode)` maps an out-of-range index to a source index,
matching SciPy exactly:
- `reflect` (half-sample): `i<0 → −i−1`, `i≥n → 2n−i−1`.
- `mirror` (whole-sample): `i<0 → −i`, `i≥n → 2n−2−i`.
- `nearest`: clamp to `[0, n−1]`.
- `wrap`: `((i mod n) + n) mod n`.
- `constant`: out-of-range samples take `cval`.

Default mode is `reflect` (SciPy's filter default).

## Filters

- **correlate1d(input, weights, axis, mode)** — slides the 1-D `weights` (origin at
  the center) along `axis`; `convolve1d` reverses the weights. The N-D
  `correlate`/`convolve` slide a 2-D kernel.
- **uniform_filter / gaussian_filter** — separable: apply `*_filter1d` along each
  axis in turn. The Gaussian 1-D kernel uses radius `int(truncate·σ + 0.5)` and the
  normalized `exp(−x²/2σ²)` weights (SciPy's `_gaussian_kernel1d`). These route
  through a backend dispatch (`last_backend()`); the CPU kernel is always present.
- **median/minimum/maximum_filter** — sliding window (square `size`) reductions with
  the boundary mode.
- **sobel/prewitt** — the separable derivative (`[−1,0,1]` smoothed by `[1,2,1]` /
  `[1,1,1]`); **laplace** — sum of second differences along each axis.

## Morphology

- **binary_erosion/dilation** — slide the structuring element (default 3×3 cross,
  connectivity 1); erosion = AND over the element, dilation = OR; `opening` =
  erode-then-dilate, `closing` = dilate-then-erode; `iterations` repeats. Border
  values follow SciPy (`border_value=0` for erosion).
- **grey_erosion/dilation** — min/max over the flat structuring element.
- **distance_transform_edt** — exact Euclidean distance to the nearest zero, via
  the Felzenszwalb–Huttenlocher separable squared-distance transform (1-D lower
  envelope of parabolas applied per axis); matches SciPy's exact EDT.

## Measurements

- **label** — connected components (default connectivity 1 / 4-neighbour) by
  union-find over nonzero pixels, then relabel in raster-scan first-appearance order
  (SciPy's labeling), returning `(labels, num_features)`.
- **center_of_mass / sum_labels / mean / maximum / minimum** — reductions over the
  whole array or per integer label.
- **find_objects** — bounding-box slices per label.

## Geometric transforms

`map_coordinates(input, coords, order, mode, cval)` samples the input at the given
coordinates with nearest (`order=0`) or bilinear (`order=1`) interpolation and the
boundary mode. `affine_transform`, `shift`, `zoom` and `rotate` are expressed
through it: each output coordinate maps back to an input coordinate via the affine
(`output = matrix·input + offset` inverse convention SciPy uses), then samples.
`zoom` uses SciPy's `(in−1)/(out−1)` grid alignment; `rotate` builds the rotation
matrix about the image center.

## Oracle strategy

Every routine compared `allclose` (or label-equivalent) to `scipy.ndimage` on small
images: each filter across boundary modes, morphology results, the EDT, `label`
output + counts, `center_of_mass`/reductions, and the geometric transforms at
orders 0/1. Tolerances tight (`~1e-9`) since all are deterministic.

## Open questions

- `label` numbering matches SciPy's raster-scan order; if a structure differs, tests
  compare up to relabeling. Geometric transforms validate orders 0/1 only; order-3
  (SciPy's default) is deferred with the spline prefilter.
