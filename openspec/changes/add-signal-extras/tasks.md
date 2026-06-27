# Tasks — signal extras (complete scipy.signal)

## 1. Advanced spectral
- [ ] generalize segment helper; `csd`, `coherence`, `spectrogram`, `stft`, `istft`

## 2. Peak analysis
- [ ] `find_peaks` (plateau midpoint; height/distance/prominence/width filters)
- [ ] `peak_prominences`, `peak_widths`

## 3. LTI systems
- [ ] `TransferFunction`, `tf2ss`, `freqresp`, `bode`
- [ ] `lsim` (ZOH via `expm`), `step`, `impulse`

## 4. Resampling
- [ ] `resample` (FFT), `upfirdn`, `resample_poly`, `decimate`

## 5. ellip / bessel design
- [ ] elliptic integral `K` + Jacobi `sn/cn/dn`; `ellipap`; `ellip`
- [ ] `besselap` (reverse Bessel polynomial roots); `bessel`

## 6. Other filtering
- [ ] `savgol_coeffs`/`savgol_filter`, `medfilt`, `wiener`
- [ ] `convolve2d`, `correlate2d`

## 7. Oracle + validation
- [ ] Extend `tests/oracle/generate.py`; regenerate
- [ ] `tests/test_signal_extras.cpp` for all of the above vs SciPy
- [ ] CPU build green; full suite green; `openspec validate add-signal-extras --strict`
- [ ] Update README
