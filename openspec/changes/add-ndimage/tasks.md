# Tasks — ndimage (Phase 11)

## 1. Module scaffold + boundary modes
- [ ] `include/scypp/ndimage/ndimage.hpp`; `src/ndimage/*.cpp` in `src/CMakeLists.txt`; export from `scypp/scypp.hpp`
- [ ] `boundary_index(i, n, mode)` for reflect/mirror/nearest/wrap/constant

## 2. Filters
- [ ] `correlate1d`/`convolve1d`, `correlate`/`convolve`
- [ ] `uniform_filter`(`1d`), `gaussian_filter`(`1d`) — separable + backend dispatch
- [ ] `median_filter`, `minimum_filter`, `maximum_filter`; `sobel`, `prewitt`, `laplace`

## 3. Morphology
- [ ] `binary_erosion`/`dilation`/`opening`/`closing`, `grey_erosion`/`dilation`
- [ ] `distance_transform_edt` (Felzenszwalb–Huttenlocher)

## 4. Measurements
- [ ] `label` (union-find, raster relabel), `center_of_mass`, `sum_labels`/`mean`/`maximum`/`minimum`, `find_objects`

## 5. Geometric transforms
- [ ] `map_coordinates` (order 0/1, modes); `affine_transform`, `shift`, `zoom`, `rotate`

## 6. Oracle + validation
- [ ] Extend `tests/oracle/generate.py`; regenerate
- [ ] `tests/test_ndimage.cpp`: filters across modes + dispatch, morphology, EDT, label/measurements, geometric transforms
- [ ] CPU build green; full suite green; `openspec validate add-ndimage --strict`
- [ ] Check off Phase 11 in `bootstrap-scypp-foundation/tasks.md`; update README
