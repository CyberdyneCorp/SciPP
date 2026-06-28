# Add linprog + NNLS to optimize

## Why

`scipy.optimize.linprog` (linear programming) and `scipy.optimize.nnls`
(nonnegative least squares) are two of the most-used optimize entry points and
were deferred from Phase 4. Both are pure-CPU classical algorithms that need no
NumPP changes, so they are high value-per-effort backlog drawdown.

## What changes

Extends the **optimize** capability — `scipp::optimize`, validated against the
SciPy oracle:

- **`linprog(c, A_ub, b_ub, A_eq, b_eq)`**: minimize `c·x` subject to
  `A_ub·x <= b_ub`, `A_eq·x = b_eq`, `x >= 0` (default bounds). Two-phase primal
  simplex with Bland's rule for anti-cycling. Reports optimal/infeasible/unbounded
  via a `status` code matching SciPy's convention (0/2/3).
- **`nnls(A, b)`**: Lawson-Hanson active-set solver for `min ||A·x - b||` with
  `x >= 0`, returning the solution and residual norm.

## Impact

- Affected specs: **modifies** the `optimize` capability (adds two requirements).
- Affected code: new `src/optimize/linprog.cpp`, `src/optimize/nnls.cpp`,
  header decls in `include/scipp/optimize/optimize.hpp`, `tests/test_optimize_lp.cpp`,
  extended oracle generator.
- Trims the `linprog`/`nnls` items from the `add-optimize-advanced` backlog.

## Non-goals

- General variable bounds beyond `x >= 0` (finite lower/upper bounds) — model via
  explicit `A_ub` rows for now.
- Interior-point / HiGHS-exact vertex selection on degenerate problems; the
  simplex matches SciPy on the optimum value and on non-degenerate vertices.
- `milp` (mixed-integer) and `linear_sum_assignment`.
