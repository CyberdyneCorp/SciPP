# Tasks — signal extras (complete scipy.signal)

## 1. Advanced spectral
- [x] generalize segment helper; `csd`, `coherence`, `spectrogram`, `stft`, `istft`

## 2. Peak analysis
- [x] `find_peaks` (plateau midpoint; height/distance/prominence/width filters)
- [x] `peak_prominences`, `peak_widths`

## 3. LTI systems
- [x] `TransferFunction`, `tf2ss`, `freqresp`, `bode`
- [x] `lsim` (ZOH via `expm`), `step`, `impulse`

## 4. Resampling
- [x] `resample` (FFT), `upfirdn`, `resample_poly`, `decimate`

## 5. ellip / bessel design
- [x] elliptic integral `K` + Jacobi `sn/cn/dn`; `ellipap`; `ellip`
- [x] `besselap` (reverse Bessel polynomial roots); `bessel`

## 6. Other filtering
- [x] `savgol_coeffs`/`savgol_filter`, `medfilt`, `wiener`
- [x] `convolve2d`, `correlate2d`

## 7. Oracle + validation
- [x] Extend `tests/oracle/generate.py`; regenerate
- [x] `tests/test_signal_extras.cpp` for all of the above vs SciPy
- [x] CPU build green; full suite green; `openspec validate add-signal-extras --strict`
- [x] Update README
