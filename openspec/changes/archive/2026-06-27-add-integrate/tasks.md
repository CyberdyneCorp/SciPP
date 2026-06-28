# Tasks — integrate + differentiate (Phase 5)

## 1. Module scaffold
- [x] `include/scipp/integrate/integrate.hpp` and `include/scipp/differentiate/differentiate.hpp` (callable typedefs + result structs); `src/integrate/*.cpp` in `src/CMakeLists.txt`; export from `scipp/scipp.hpp`

## 2. Fixed-sample quadrature
- [x] `trapezoid`, `simpson` (even-interval correction), `cumulative_trapezoid`

## 3. Adaptive / fixed quadrature
- [x] `fixed_quad` (Gauss–Legendre nodes/weights via Newton on Legendre polys)
- [x] `quad` (adaptive Gauss–Kronrod 21/10, heap-based interval subdivision, value + abserr)

## 4. ODE integration
- [x] RK45 (Dormand–Prince) and RK23 (Bogacki–Shampine) embedded-RK steps
- [x] Adaptive step controller (scaled RMS error, step clip), SciPy initial-step heuristic
- [x] `solve_ivp` with `t_eval` (step capped onto eval points), `OdeResult`

## 5. Differentiation
- [x] `derivative` (central differences + Richardson extrapolation)
- [x] `jacobian`, `hessian` (central differences)

## 6. Oracle + validation
- [x] Extend `tests/oracle/generate.py` with quadrature golden values; regenerate
- [x] `tests/test_integrate.cpp`: quadrature vs SciPy + analytic; solve_ivp vs analytic (RK45/RK23); derivative/jacobian/hessian vs closed form
- [x] CPU build green; full suite green
- [x] `openspec validate add-integrate --strict` green
- [x] Check off Phase 5 in `bootstrap-scipp-foundation/tasks.md`; update README
