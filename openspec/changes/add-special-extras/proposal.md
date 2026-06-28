# Special-function extras (deferred backlog)

## Why

Phase 1 (`add-special-constants`, archived) delivered the special-function core —
the gamma/erf/Bessel families, orthogonal-polynomial evaluators, and
`logsumexp`/`softmax`. It explicitly deferred the rest of `scipy.special`, but —
unlike every other module — that deferred surface was never captured in a backlog
change. This change records it so the high-use families can be scheduled.

All of these are pure-CPU scalar/elementwise kernels (Cephes / continued-fraction
/ series algorithms, in the same style as the existing `gammainc`/`erfinv`) and
need **no NumPP changes**.

## What changes

Adds (as target requirements) to the **special** capability, validated against the
SciPy oracle:

- **Airy**: `airy`, `airye`.
- **Elliptic integrals**: `ellipk`, `ellipkm1`, `ellipe`, `ellipkinc`, `ellipeinc`,
  `ellipj`.
- **Error-function relatives**: `erfcx`, `dawsn`, `wofz` / `voigt_profile`,
  `fresnel`.
- **More Bessel**: `spherical_jn`/`yn`/`in`/`kn`, `kelvin`, `hankel1`/`hankel2`.
- **Integrals**: `sici`, `shichi`.
- **Misc special**: `lambertw`, `zeta`/`zetac`, `struve`/`modstruve`, `spence`.
- **Hypergeometric**: `hyp0f1`, `hyp1f1`, `hyp2f1`, `hyperu`.

## Non-goals

- The long tail beyond the above: Mathieu, spheroidal-wave, parabolic-cylinder,
  and the full multi-precision hypergeometric edge cases — these stay deferred.
- Complex-argument variants beyond what the listed functions need.
- Implementing anything in this change itself; it is a tracking artifact. Each
  family graduates into its own focused change (e.g. `add-special-airy-elliptic`).
