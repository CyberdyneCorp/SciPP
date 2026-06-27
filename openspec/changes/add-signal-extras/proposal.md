# Complete scipy.signal (signal extras)

## Why

Phase 8 (`add-signal`, archived) delivered the DSP core but deferred several
**widely-used** subsystems — advanced spectral analysis, peak finding, LTI system
objects, resampling, the remaining IIR families, and 2-D/statistical filters.
These are everyday tools, not edge cases, so this change completes
`scipy.signal`'s commonly-used surface. It builds directly on Phase 8 (the
`_spectral` helper, `cheby1`, `filtfilt`, `freqz`) and on `scypp::linalg`
(`expm`/`lstsq` for state-space simulation and Savitzky–Golay).

## What changes

Extends the **signal** capability — `scypp::signal`, validated against the SciPy
oracle:

- **Advanced spectral**: `csd`, `coherence`, `spectrogram`, `stft`, `istft`.
- **Peak analysis**: `find_peaks` (with `height`/`distance`/`prominence`/`width`
  filters), `peak_prominences`, `peak_widths`.
- **LTI systems**: `TransferFunction`, `tf2ss`, `freqresp`, `bode`, `step`,
  `impulse`, `lsim` (continuous-time, ZOH simulation via `expm`).
- **Resampling**: `resample` (FFT), `resample_poly`, `decimate`, `upfirdn`.
- **Filter design**: `ellip` (elliptic) and `bessel` (Bessel/Thomson), with the
  analog prototypes `ellipap`/`besselap` and the supporting elliptic integrals /
  Jacobi functions.
- **Other filtering**: `savgol_filter`/`savgol_coeffs`, `medfilt`, `wiener`,
  `convolve2d`, `correlate2d`.

## Impact

- Affected specs: extends the `signal` capability (merged into the baseline on
  archive).
- Affected code: new `src/signal/*` translation units, additions to
  `include/scypp/signal/signal.hpp`, `tests/test_signal_extras.cpp`, extended
  oracle generator.

## Non-goals

- 2-D filter design (`sepfir2d`), `ShortTimeFFT` class API, `vectorstrength`,
  `chirp` extras, and the long tail of windows beyond Phase 8 — separate if needed.
- `lsim`/`step`/`impulse` for discrete-time `dlti` systems (continuous-time only
  here).
