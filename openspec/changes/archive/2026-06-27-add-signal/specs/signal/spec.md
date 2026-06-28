# signal Specification

## ADDED Requirements

### Requirement: Convolution and correlation

`scipp::signal` SHALL provide `convolve`, `correlate` and `fftconvolve` with modes
`"full"`, `"same"` and `"valid"`, matching SciPy. (oracle: scipy/signal/_signaltools.py)

#### Scenario: Convolution modes match SciPy
- GIVEN two sequences
- WHEN `convolve` and `correlate` are computed in each mode
- THEN the results are `allclose` to SciPy's, and `fftconvolve` matches the direct
  `convolve` within floating-point tolerance

### Requirement: Window functions

`scipp::signal` SHALL provide `get_window` and the windows `boxcar`, `hann`,
`hamming`, `blackman`, `bartlett`, `blackmanharris`, `flattop`, `kaiser` and
`tukey`, matching SciPy (including the symmetric/periodic distinction). (oracle:
scipy/signal/windows)

#### Scenario: Windows match SciPy
- GIVEN a length `M`
- WHEN each window is generated (symmetric and periodic)
- THEN the coefficients are `allclose` to SciPy's, and `get_window(name, M)`
  dispatches to the same result

### Requirement: Time-domain filtering

`scipp::signal` SHALL provide `lfilter`, `lfilter_zi`, `filtfilt`, `sosfilt`,
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

`scipp::signal` SHALL provide IIR design `butter`, `cheby1`, `cheby2`
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

`scipp::signal` SHALL provide `periodogram` and `welch`, returning frequencies and
the power spectral density, matching SciPy. (oracle: scipy/signal/_spectral_py.py)

#### Scenario: Periodogram and Welch match SciPy
- GIVEN a sampled signal and a sampling frequency
- WHEN `periodogram(x, fs)` and `welch(x, fs)` are computed
- THEN the returned frequencies and PSD are `allclose` to SciPy's

### Requirement: Standard waveforms

`scipp::signal` SHALL provide `chirp`, `sawtooth`, `square` and `unit_impulse`,
matching SciPy. (oracle: scipy/signal/_waveforms.py)

#### Scenario: Waveforms match SciPy
- GIVEN a time vector and parameters
- WHEN each waveform is generated
- THEN the samples are `allclose` to SciPy's
