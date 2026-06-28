# Add ndimage (Phase 11)

## Why

Phase 11 of the SciPP roadmap. `scipy.ndimage` provides N-dimensional image
processing — filtering, morphology, measurements, and geometric transforms. Its
**separable convolution** (Gaussian/uniform filters) is a GPU acceleration target,
routed like the Phase-9/10 kernels through NumPP's capability registry.

`scipy.ndimage` is large; this change delivers the deterministic core (the
separable and rank filters, binary/greyscale morphology, the exact distance
transform, connected-component measurements, and order-0/1 geometric transforms)
and defers cubic-spline (order ≥ 2) interpolation and the Fourier filters.

## What changes

Adds the **ndimage** capability — `scipp::ndimage`, validated against the SciPy
oracle:

- **Filters**: `correlate1d`/`convolve1d`, `correlate`/`convolve`,
  `uniform_filter`(`1d`), `gaussian_filter`(`1d`), `median_filter`,
  `minimum_filter`, `maximum_filter`, `sobel`, `prewitt`, `laplace`, with the
  boundary modes `reflect`/`nearest`/`mirror`/`wrap`/`constant`; separable filters
  route through a backend dispatch.
- **Morphology**: `binary_erosion`/`dilation`/`opening`/`closing`,
  `grey_erosion`/`dilation`, `distance_transform_edt` (exact Euclidean).
- **Measurements**: `label` (connected components), `center_of_mass`,
  `sum_labels`/`mean`/`maximum`/`minimum`, `find_objects`.
- **Geometric transforms**: `shift`, `zoom`, `rotate`, `affine_transform`,
  `map_coordinates` (interpolation orders 0 and 1).

## Impact

- Affected specs: **adds** the `ndimage` capability.
- Affected code: new `include/scipp/ndimage/`, `src/ndimage/`,
  `tests/test_ndimage.cpp`, extended oracle generator.
- Roadmap: checks off Phase 11 in `bootstrap-scipp-foundation/tasks.md`.

## Non-goals (deferred)

- **Cubic-spline interpolation** (order ≥ 2) for the geometric transforms and
  `spline_filter` — needs the spline prefilter; orders 0/1 ship here.
- **Fourier filters**: `fourier_gaussian`/`fourier_uniform`/`fourier_shift`/
  `fourier_ellipsoid`.
- **Rank/percentile filters** beyond min/median/max, `rank_filter`,
  `percentile_filter`, `gaussian_laplace`, `gaussian_gradient_magnitude`.
- **Morphology extras**: `binary_fill_holes`, `binary_hit_or_miss`,
  `distance_transform_cdt`/`_bf`, `watershed_ift`, `grey_opening`/`closing`,
  `morphological_gradient`.
- **Actual GPU separable-convolution device kernel** — ships the dispatch
  architecture + CPU kernel; the device kernel lands with a NumPP backend (tracked).
