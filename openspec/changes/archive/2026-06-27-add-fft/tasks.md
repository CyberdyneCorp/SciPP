# Tasks — fft + fftpack (Phase 3)

## 1. Module scaffold
- [x] `include/scypp/fft/fft.hpp` declarations; `src/fft/*.cpp` in `src/CMakeLists.txt`; export from `scypp/scypp.hpp`

## 2. FFTs (delegate to NumPP)
- [x] `fft`/`ifft`, `rfft`/`irfft`, `hfft`/`ihfft` (n, axis, norm)
- [x] `fft2`/`ifft2`, `fftn`/`ifftn`, `rfftn`/`irfftn` (s, axes, norm)

## 3. DCT/DST (implement)
- [x] `dct`/`dst` types I–IV, default `norm=None`, direct summation kernels
- [x] `idct`/`idst` via inverse-type relations + scale (verified vs oracle)
- [x] `apply_axis` helper for axis-wise transforms

## 4. Helpers + fftpack
- [x] `fftfreq`, `rfftfreq`, `fftshift`, `ifftshift` (delegate)
- [x] `next_fast_len` (smallest 11-smooth ≥ n)
- [x] `scypp::fftpack` namespace re-exporting fft/dct/dst entry points

## 5. Oracle + validation
- [x] Extend `tests/oracle/generate.py` with fft/dct/dst golden data; regenerate
- [x] `tests/test_fft.cpp`: `allclose` to SciPy, `ifft(fft)`/`idct(dct)` round-trips, axis behavior, next_fast_len, fftpack==fft
- [x] CPU build green; full suite green
- [x] `openspec validate add-fft --strict` green
- [x] Check off Phase 3 in `bootstrap-scypp-foundation/tasks.md`; update README
