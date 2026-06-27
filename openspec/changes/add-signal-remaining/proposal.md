# Signal remaining (deferred backlog)

## Why

The `signal` capability is largely complete (Phase 8 + signal-extras). This change
tracks the small remaining tail. Not implemented yet.

## What changes

Adds (as target requirements) to the **signal** capability:

- **Discrete-time LTI** (`dlti`): `dstep`, `dimpulse`, `dlsim`, `dbode`,
  `dfreqresp`, and `cont2discrete`.
- **2-D separable filtering**: `sepfir2d`.
- **Misc**: `ShortTimeFFT` class API, `vectorstrength`, `peak_widths` rel-height
  variants, and `place_poles`.

## Non-goals
- Implementing anything here; tracking only.
