# Add signal (Phase 8)

## Why

Phase 8 of the SciPP roadmap. `scipy.signal` is the digital-signal-processing
toolbox — convolution, filtering, filter design, and spectral analysis. It builds
on the FFT from Phase 3 (for FFT convolution, the analytic signal, and Welch
spectra) and on `numpp` array operations.

`scipy.signal` is the largest single subpackage; this change delivers the
deterministic DSP core (convolution, windows, time-domain filtering, IIR/FIR
design, spectral estimation, and standard waveforms) and defers the stateful LTI
system objects, resampling, peak finding, and 2-D filtering.

## What changes

Adds the **signal** capability — `scipp::signal`, validated against the SciPy
oracle:

- **Convolution / correlation**: `convolve`, `correlate`, `fftconvolve`
  (modes `full`/`same`/`valid`).
- **Windows**: `get_window` plus `boxcar`, `hann`, `hamming`, `blackman`,
  `bartlett`, `blackmanharris`, `flattop`, `kaiser`, `tukey`.
- **Time-domain filtering**: `lfilter`, `lfilter_zi`, `filtfilt` (zero-phase),
  `sosfilt`, `detrend`, `hilbert` (analytic signal), `freqz`.
- **Filter design**: IIR `butter`/`cheby1`/`cheby2` (lowpass/highpass/bandpass/
  bandstop via analog prototype + bilinear transform), FIR `firwin`, and the
  representation conversions `tf2zpk`/`zpk2tf`/`zpk2sos`/`tf2sos`.
- **Spectral analysis**: `periodogram` and `welch`.
- **Waveforms**: `chirp`, `sawtooth`, `square`, `unit_impulse`.

## Impact

- Affected specs: **adds** the `signal` capability.
- Affected code: new `include/scipp/signal/`, `src/signal/`,
  `tests/test_signal.cpp`, extended oracle generator. Uses `scipp::fft`/`numpp::fft`
  for FFT-based routines.
- Roadmap: checks off Phase 8 in `bootstrap-scipp-foundation/tasks.md`.

## Non-goals (deferred)

- **LTI system objects**: `lti`, `TransferFunction`, `StateSpace`, `bode`, `step`,
  `impulse`, `lsim`, `freqresp`.
- **Resampling**: `resample`, `resample_poly`, `decimate`, `upfirdn`.
- **Peak analysis**: `find_peaks`, `peak_widths`, `argrelextrema`.
- **Advanced spectral**: `spectrogram`, `stft`/`istft`, `csd`, `coherence`.
- **Other filter design**: `ellip`, `bessel`, `iirdesign`, `remez`, `savgol_filter`,
  `medfilt`, `wiener`.
- **2-D**: `convolve2d`, `correlate2d`, `sepfir2d`.
