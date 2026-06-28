# Add Powell, CG, and L-BFGS-B methods to minimize

## Why

`scypp::optimize::minimize` shipped with only Nelder-Mead and BFGS. Powell, CG,
and L-BFGS-B are the next most-used `scipy.optimize.minimize` methods and cover
the derivative-free, nonlinear conjugate-gradient, and bound-constrained
limited-memory cases respectively. L-BFGS-B in particular unlocks the first
box-constrained minimizer in the library. All three are classical CPU algorithms
needing no NumPP changes.

## What changes

Extends the **optimize** capability — `scypp::optimize::minimize`, validated
against the SciPy oracle:

- **`method="Powell"`**: derivative-free direction-set method. Each iteration
  line-minimizes along every basis direction with a downhill-bracket + Brent
  scalar minimizer, then replaces the direction of largest decrease with the net
  move (with Powell's extrapolation test).
- **`method="CG"`**: nonlinear conjugate gradient with Polak-Ribiere+ `beta`,
  central-difference numerical gradient, and a strong-Wolfe line search.
- **`method="L-BFGS-B"`**: limited-memory BFGS two-loop recursion with optional
  box bounds enforced by gradient projection (clamping trial points to
  `[lo, hi]`). The unbounded call also works.

The `minimize` signature gains a trailing optional `bounds` argument
(`std::optional<Bounds>`, one `(lo, hi)` pair per coordinate); existing callers
compile unchanged.

## Impact

- Affected specs: **modifies** the `optimize` capability (one new requirement).
- Affected code: `src/optimize/minimize.cpp` (new methods + dispatch),
  `include/scypp/optimize/optimize.hpp` (`Bounds` alias, `bounds` parameter),
  `tests/test_optimize.cpp`, extended oracle generator.
- Trims the delivered methods from the `add-optimize-advanced` backlog.

## Non-goals

- `Newton-CG`, `TNC`, and the trust-region family — still deferred in the
  `add-optimize-advanced` backlog.
- Matching SciPy's exact iteration counts / `nfev`; only the optimum `x`
  (to ~1e-4) and `fun` (to ~1e-6) are validated.
- L-BFGS-B's exact generalized-Cauchy-point subspace minimization; the gradient
  projection variant matches SciPy on the bounded and unbounded optima used here.
