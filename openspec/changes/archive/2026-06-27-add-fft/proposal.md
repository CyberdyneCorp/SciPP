# Add fft + fftpack (Phase 3)

## Why

Phase 3 of the ScyPP roadmap. `scipy.fft` is the spectral-analysis backbone for
`signal`, and its discrete cosine/sine transforms (DCT/DST) are needed across
signal processing and compression. NumPP already provides the complex/real FFT
core (`fft`/`ifft`/`rfft`/`irfft`/`hfft`/`ihfft`, the N-D variants, and
`fftfreq`/`fftshift`) with the **exact SciPy `norm` conventions**
(`backward`/`ortho`/`forward`). ScyPP therefore delegates the FFTs and **adds the
SciPy-only DCT/DST family** plus the `next_fast_len` helper and the legacy
`fftpack` alias namespace.

## What changes

Adds the **fft** capability — `scypp::fft` (and a thin `scypp::fftpack`):

- **FFTs (delegate to NumPP, identical signatures)**: `fft`/`ifft`,
  `rfft`/`irfft`, `hfft`/`ihfft`, `fft2`/`ifft2`, `fftn`/`ifftn`,
  `rfftn`/`irfftn`, with `n`/`s`, `axis`/`axes` and `norm`.
- **DCT/DST (implemented)**: `dct`/`idct` and `dst`/`idst`, types I–IV, with the
  default `norm=None` ("backward") normalization, applied along a selectable axis.
- **Helpers**: `fftfreq`, `rfftfreq`, `fftshift`, `ifftshift` (delegate), and
  `next_fast_len` (smallest 11-smooth length ≥ n, implemented).
- **fftpack**: a `scypp::fftpack` namespace re-exporting `fft`/`ifft`/`rfft`/
  `irfft`/`dct`/`idct`/`dst`/`idst` for the legacy API surface.

## Impact

- Affected specs: **adds** the `fft` capability.
- Affected code: new `include/scypp/fft/`, `src/fft/`, `tests/test_fft.cpp`,
  extended oracle generator. Reuses the Phase 1–2 foundation unchanged.
- Roadmap: checks off Phase 3 in `bootstrap-scypp-foundation/tasks.md`.

## Non-goals (deferred)

- **`ortho`/`forward` normalization for DCT/DST** — only the default `norm=None`
  ("backward") is implemented this phase; the FFT family still supports all three
  via NumPP.
- **N-D DCT/DST** (`dctn`/`idctn`/`dstn`/`idstn`) and multi-axis cosine transforms.
- **`fht`/`ifht`** (Hankel) and **`czt`** (chirp-z) transforms.
- **`workers=` parallelism** and FFT plan caching.
