# signal Specification

## Purpose
TBD - created by archiving change add-signal. Update Purpose after archive.
## Requirements
### Requirement: Convolution and correlation

`scypp::signal` SHALL provide `convolve`, `correlate` and `fftconvolve` with modes
`"full"`, `"same"` and `"valid"`, matching SciPy. (oracle: scipy/signal/_signaltools.py)

#### Scenario: Convolution modes match SciPy
- GIVEN two sequences
- WHEN `convolve` and `correlate` are computed in each mode
- THEN the results are `allclose` to SciPy's, and `fftconvolve` matches the direct
  `convolve` within floating-point tolerance

### Requirement: Window functions

`scypp::signal` SHALL provide `get_window` and the windows `boxcar`, `hann`,
`hamming`, `blackman`, `bartlett`, `blackmanharris`, `flattop`, `kaiser` and
`tukey`, matching SciPy (including the symmetric/periodic distinction). (oracle:
scipy/signal/windows)

#### Scenario: Windows match SciPy
- GIVEN a length `M`
- WHEN each window is generated (symmetric and periodic)
- THEN the coefficients are `allclose` to SciPy's, and `get_window(name, M)`
  dispatches to the same result

### Requirement: Time-domain filtering

`scypp::signal` SHALL provide `lfilter`, `lfilter_zi`, `filtfilt`, `sosfilt`,
`detrend`, `hilbert` and `freqz`, matching SciPy. `filtfilt` SHALL apply zero-phase
filtering with SciPy's default odd padding. (oracle: scipy/signal/_signaltools.py)

#### Scenario: lfilter and sosfilt match SciPy
- GIVEN filter coefficients and an input signal
- WHEN `lfilter(b, a, x)` and the equivalent `sosfilt(sos, x)` are computed
- THEN both are `allclose` to SciPy's outputs

#### Scenario: Zero-phase filtfilt
- GIVEN a filter and a signal
- WHEN `filtfilt(b, a, x)` is computed
- THEN the result is `allclose` to SciPy's and has zero phase distortion
  (symmetric response to a symmetric input)

#### Scenario: Analytic signal and frequency response
- GIVEN a real signal and a designed filter
- WHEN `hilbert(x)` and `freqz(b, a)` are computed
- THEN the analytic signal and the complex frequency response are `allclose` to
  SciPy's

### Requirement: Filter design

`scypp::signal` SHALL provide IIR design `butter`, `cheby1`, `cheby2`
(lowpass/highpass/bandpass/bandstop), FIR design `firwin`, and the representation
conversions `tf2zpk`, `zpk2tf`, `zpk2sos` and `tf2sos`, matching SciPy. (oracle:
scipy/signal/_filter_design.py)

#### Scenario: IIR design matches SciPy
- GIVEN an order, critical frequency and band type
- WHEN `butter`/`cheby1`/`cheby2` produce `(b, a)` (and `sos`)
- THEN the coefficients are `allclose` to SciPy's and the `freqz` response matches

#### Scenario: FIR design matches SciPy
- GIVEN a number of taps and cutoff
- WHEN `firwin` is computed
- THEN the taps are `allclose` to SciPy's

#### Scenario: Representation conversions round-trip
- GIVEN a transfer function `(b, a)`
- WHEN converted via `tf2zpk`/`zpk2tf` (and `zpk2sos`)
- THEN the round-trip reconstructs `(b, a)` within tolerance and matches SciPy

### Requirement: Spectral estimation

`scypp::signal` SHALL provide `periodogram` and `welch`, returning frequencies and
the power spectral density, matching SciPy. (oracle: scipy/signal/_spectral_py.py)

#### Scenario: Periodogram and Welch match SciPy
- GIVEN a sampled signal and a sampling frequency
- WHEN `periodogram(x, fs)` and `welch(x, fs)` are computed
- THEN the returned frequencies and PSD are `allclose` to SciPy's

### Requirement: Standard waveforms

`scypp::signal` SHALL provide `chirp`, `sawtooth`, `square` and `unit_impulse`,
matching SciPy. (oracle: scipy/signal/_waveforms.py)

#### Scenario: Waveforms match SciPy
- GIVEN a time vector and parameters
- WHEN each waveform is generated
- THEN the samples are `allclose` to SciPy's

### Requirement: Advanced spectral analysis

`scypp::signal` SHALL provide `csd`, `coherence`, `spectrogram`, `stft` and
`istft`, matching SciPy. (oracle: scipy/signal/_spectral_py.py)

#### Scenario: Cross-spectral density and coherence
- GIVEN two signals
- WHEN `csd(x, y)` and `coherence(x, y)` are computed
- THEN the frequencies, cross-spectral density and coherence are `allclose` to
  SciPy's

#### Scenario: Spectrogram and STFT round-trip
- GIVEN a signal
- WHEN `spectrogram(x)` and `stft(x)` are computed
- THEN their frequencies/times and the spectral matrices are `allclose` to SciPy's,
  and `istft(stft(x))` reconstructs `x` within tolerance

### Requirement: Peak analysis

`scypp::signal` SHALL provide `find_peaks` (with `height`, `distance`,
`prominence` and `width` filters), `peak_prominences` and `peak_widths`, matching
SciPy. (oracle: scipy/signal/_peak_finding.py)

#### Scenario: Find peaks with filters
- GIVEN a signal with several local maxima
- WHEN `find_peaks` is called with and without `height`/`distance`/`prominence`
- THEN the returned peak indices match SciPy's

#### Scenario: Prominences and widths
- GIVEN a signal and its peaks
- WHEN `peak_prominences` and `peak_widths` are computed
- THEN the prominences and widths are `allclose` to SciPy's

### Requirement: LTI system analysis

`scypp::signal` SHALL provide a continuous-time `TransferFunction` with
`freqresp`, `bode`, `step`, `impulse` and `lsim`, matching SciPy. (oracle:
scipy/signal/_ltisys.py)

#### Scenario: Frequency response and Bode
- GIVEN a transfer function
- WHEN `freqresp` and `bode` are computed
- THEN the complex response, magnitude (dB) and phase (deg) are `allclose` to
  SciPy's

#### Scenario: Time-domain responses
- GIVEN a stable transfer function
- WHEN `step`, `impulse` and `lsim` are computed at given times
- THEN the responses are `allclose` to SciPy's

### Requirement: Resampling

`scypp::signal` SHALL provide `resample`, `resample_poly`, `decimate` and
`upfirdn`, matching SciPy. (oracle: scipy/signal/_signaltools.py)

#### Scenario: FFT and polyphase resampling
- GIVEN a signal
- WHEN `resample(x, num)`, `resample_poly(x, up, down)` and `decimate(x, q)` are
  computed
- THEN the resampled signals are `allclose` to SciPy's

#### Scenario: upfirdn
- GIVEN a filter and a signal
- WHEN `upfirdn(h, x, up, down)` is computed
- THEN the result is `allclose` to SciPy's

### Requirement: Elliptic and Bessel filter design

`scypp::signal` SHALL provide `ellip` and `bessel` filter design (lowpass/
highpass/bandpass/bandstop), matching SciPy within documented tolerance. (oracle:
scipy/signal/_filter_design.py)

#### Scenario: ellip and bessel match SciPy
- GIVEN an order, ripple specifications and critical frequency
- WHEN `ellip`/`bessel` produce `(b, a)`
- THEN the coefficients (or the realized `freqz` response) are `allclose` to
  SciPy's

### Requirement: Savitzky–Golay, median, Wiener, and 2-D filtering

`scypp::signal` SHALL provide `savgol_coeffs`/`savgol_filter`, `medfilt`,
`wiener`, `convolve2d` and `correlate2d`, matching SciPy. (oracle:
scipy/signal/_savitzky_golay.py, _signaltools.py)

#### Scenario: Savitzky–Golay smoothing
- GIVEN a noisy signal, window length and polynomial order
- WHEN `savgol_filter` is applied
- THEN the result is `allclose` to SciPy's, and `savgol_coeffs` matches

#### Scenario: Median, Wiener, and 2-D convolution
- GIVEN appropriate inputs
- WHEN `medfilt`, `wiener`, `convolve2d` and `correlate2d` are computed
- THEN the outputs are `allclose` to SciPy's

