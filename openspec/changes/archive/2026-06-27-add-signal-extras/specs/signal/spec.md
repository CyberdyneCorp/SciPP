# signal Specification

## ADDED Requirements

### Requirement: Advanced spectral analysis

`scipp::signal` SHALL provide `csd`, `coherence`, `spectrogram`, `stft` and
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

`scipp::signal` SHALL provide `find_peaks` (with `height`, `distance`,
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

`scipp::signal` SHALL provide a continuous-time `TransferFunction` with
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

`scipp::signal` SHALL provide `resample`, `resample_poly`, `decimate` and
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

`scipp::signal` SHALL provide `ellip` and `bessel` filter design (lowpass/
highpass/bandpass/bandstop), matching SciPy within documented tolerance. (oracle:
scipy/signal/_filter_design.py)

#### Scenario: ellip and bessel match SciPy
- GIVEN an order, ripple specifications and critical frequency
- WHEN `ellip`/`bessel` produce `(b, a)`
- THEN the coefficients (or the realized `freqz` response) are `allclose` to
  SciPy's

### Requirement: Savitzky–Golay, median, Wiener, and 2-D filtering

`scipp::signal` SHALL provide `savgol_coeffs`/`savgol_filter`, `medfilt`,
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
