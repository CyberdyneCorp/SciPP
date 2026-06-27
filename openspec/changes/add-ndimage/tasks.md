# Tasks — ndimage (Phase 11)

## 1. Module scaffold + boundary modes
- [x] `include/scypp/ndimage/ndimage.hpp`; `src/ndimage/*.cpp` in `src/CMakeLists.txt`; export from `scypp/scypp.hpp`
- [x] `boundary_index(i, n, mode)` for reflect/mirror/nearest/wrap/constant

## 2. Filters
- [x] `correlate1d`/`convolve1d`, `correlate`/`convolve`
- [x] `uniform_filter`(`1d`), `gaussian_filter`(`1d`) — separable + backend dispatch
- [x] `median_filter`, `minimum_filter`, `maximum_filter`; `sobel`, `prewitt`, `laplace`

## 3. Morphology
- [x] `binary_erosion`/`dilation`/`opening`/`closing`, `grey_erosion`/`dilation`
- [x] `distance_transform_edt` (Felzenszwalb–Huttenlocher)

## 4. Measurements
- [x] `label` (union-find, raster relabel), `center_of_mass`, `sum_labels`/`mean`/`maximum`/`minimum`, `find_objects`

## 5. Geometric transforms
- [x] `map_coordinates` (order 0/1, modes); `affine_transform`, `shift`, `zoom`, `rotate`

## 6. Oracle + validation
- [x] Extend `tests/oracle/generate.py`; regenerate
- [x] `tests/test_ndimage.cpp`: filters across modes + dispatch, morphology, EDT, label/measurements, geometric transforms
- [x] CPU build green; full suite green; `openspec validate add-ndimage --strict`
- [x] Check off Phase 11 in `bootstrap-scypp-foundation/tasks.md`; update README
