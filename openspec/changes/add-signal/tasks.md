# Tasks — signal (Phase 8)

## 1. Module scaffold
- [ ] `include/scypp/signal/signal.hpp` (result structs + decls); `src/signal/*.cpp` in `src/CMakeLists.txt`; export from `scypp/scypp.hpp`

## 2. Convolution + waveforms + windows
- [ ] `convolve`, `correlate`, `fftconvolve` (full/same/valid)
- [ ] windows: `boxcar`, `hann`, `hamming`, `blackman`, `bartlett`, `blackmanharris`, `flattop`, `kaiser`, `tukey`, `get_window`
- [ ] waveforms: `chirp`, `sawtooth`, `square`, `unit_impulse`

## 3. Time-domain filtering
- [ ] `lfilter` (DF2T), `lfilter_zi`, `sosfilt`
- [ ] `filtfilt` (odd padding, zero-phase)
- [ ] `detrend` (constant/linear), `hilbert` (analytic via FFT), `freqz`

## 4. Filter design
- [ ] analog prototypes `buttap`/`cheb1ap`/`cheb2ap`; transforms `lp2lp`/`lp2hp`/`lp2bp`/`lp2bs`; `bilinear_zpk`
- [ ] `butter`, `cheby1`, `cheby2` (lp/hp/bp/bs); `firwin`
- [ ] `tf2zpk`, `zpk2tf`, `zpk2sos`, `tf2sos`

## 5. Spectral
- [ ] `periodogram`, `welch` (density scaling, one-sided, Hann default)

## 6. Oracle + validation
- [ ] Extend `tests/oracle/generate.py` with signal golden values; regenerate
- [ ] `tests/test_signal.cpp`: convolution/windows/waveforms; lfilter/filtfilt/sosfilt; butter/cheby/firwin + freqz; welch/periodogram
- [ ] CPU build green; full suite green
- [ ] `openspec validate add-signal --strict` green
- [ ] Check off Phase 8 in `bootstrap-scypp-foundation/tasks.md`; update README
