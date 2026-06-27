# Tasks — integrate + differentiate (Phase 5)

## 1. Module scaffold
- [ ] `include/scypp/integrate/integrate.hpp` and `include/scypp/differentiate/differentiate.hpp` (callable typedefs + result structs); `src/integrate/*.cpp` in `src/CMakeLists.txt`; export from `scypp/scypp.hpp`

## 2. Fixed-sample quadrature
- [ ] `trapezoid`, `simpson` (even-interval correction), `cumulative_trapezoid`

## 3. Adaptive / fixed quadrature
- [ ] `fixed_quad` (Gauss–Legendre nodes/weights via Newton on Legendre polys)
- [ ] `quad` (adaptive Gauss–Kronrod 21/10, heap-based interval subdivision, value + abserr)

## 4. ODE integration
- [ ] RK45 (Dormand–Prince) and RK23 (Bogacki–Shampine) embedded-RK steps
- [ ] Adaptive step controller (scaled RMS error, step clip), SciPy initial-step heuristic
- [ ] `solve_ivp` with `t_eval` (step capped onto eval points), `OdeResult`

## 5. Differentiation
- [ ] `derivative` (central differences + Richardson extrapolation)
- [ ] `jacobian`, `hessian` (central differences)

## 6. Oracle + validation
- [ ] Extend `tests/oracle/generate.py` with quadrature golden values; regenerate
- [ ] `tests/test_integrate.cpp`: quadrature vs SciPy + analytic; solve_ivp vs analytic (RK45/RK23); derivative/jacobian/hessian vs closed form
- [ ] CPU build green; full suite green
- [ ] `openspec validate add-integrate --strict` green
- [ ] Check off Phase 5 in `bootstrap-scypp-foundation/tasks.md`; update README
