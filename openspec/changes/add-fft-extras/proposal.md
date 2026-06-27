# FFT extras (deferred backlog)

## Why

Phase 3 (`add-fft`, archived) delivered the FFT core and 1-D DCT/DST. This change
is the **tracked backlog** for the deferred transforms. Not implemented yet.

## What changes

Adds (as target requirements) to the **fft** capability:

- **DCT/DST `ortho` and `forward` normalizations** (only `backward`/None ship today).
- **N-D cosine/sine transforms**: `dctn`, `idctn`, `dstn`, `idstn`.
- **Hankel transform**: `fht`, `ifht`.
- **Chirp-z transform**: `czt`, `zoom_fft`, `CZT`.

## Non-goals
- Implementing anything here; tracking only.
