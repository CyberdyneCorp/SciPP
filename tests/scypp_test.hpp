#pragma once
// Minimal self-contained test harness (mirrors NumPP's numpp_test.hpp) — zero
// external dependencies so the test build works offline and on mobile
// toolchains.

#include <cmath>
#include <cstdio>
#include <exception>
#include <functional>
#include <string>
#include <vector>

namespace spt {

struct Case { std::string name; std::function<void()> fn; };

inline std::vector<Case>& registry() { static std::vector<Case> r; return r; }
inline int& failures() { static int f = 0; return f; }
inline int& checks() { static int c = 0; return c; }

struct Registrar {
  Registrar(std::string name, std::function<void()> fn) {
    registry().push_back({std::move(name), std::move(fn)});
  }
};

inline void report_fail(const char* file, int line, const std::string& msg) {
  ++failures();
  std::fprintf(stderr, "  FAIL %s:%d  %s\n", file, line, msg.c_str());
}

inline bool close(double a, double b, double rtol, double atol) {
  if (std::isnan(a) && std::isnan(b)) return true;
  if (std::isinf(a) || std::isinf(b)) return a == b;
  return std::fabs(a - b) <= atol + rtol * std::fabs(b);
}

inline int run_all() {
  int passed = 0;
  for (auto& c : registry()) {
    int before = failures();
    try {
      c.fn();
    } catch (const std::exception& e) {
      report_fail(__FILE__, __LINE__, std::string("uncaught exception: ") + e.what());
    }
    if (failures() == before) { ++passed; }
    else { std::fprintf(stderr, "[FAILED] %s\n", c.name.c_str()); }
  }
  std::fprintf(stderr, "\n%d/%zu cases passed, %d checks, %d failures\n",
               passed, registry().size(), checks(), failures());
  return failures() == 0 ? 0 : 1;
}

}  // namespace spt

#define SPT_CONCAT_(a, b) a##b
#define SPT_CONCAT(a, b) SPT_CONCAT_(a, b)
#define TEST_CASE(name)                                                        \
  static void SPT_CONCAT(spt_fn_, __LINE__)();                                 \
  static ::spt::Registrar SPT_CONCAT(spt_reg_, __LINE__){name, SPT_CONCAT(spt_fn_, __LINE__)}; \
  static void SPT_CONCAT(spt_fn_, __LINE__)()

#define CHECK(cond)                                                            \
  do {                                                                         \
    ++::spt::checks();                                                         \
    if (!(cond)) ::spt::report_fail(__FILE__, __LINE__, "CHECK(" #cond ")");   \
  } while (0)

#define CHECK_CLOSE(a, b, rtol, atol)                                          \
  do {                                                                         \
    ++::spt::checks();                                                         \
    double va = (a), vb = (b);                                                 \
    if (!::spt::close(va, vb, rtol, atol)) {                                   \
      char buf[160];                                                           \
      std::snprintf(buf, sizeof(buf), "CLOSE(%s): got %.17g want %.17g", #a, va, vb); \
      ::spt::report_fail(__FILE__, __LINE__, buf);                            \
    }                                                                          \
  } while (0)

#define CHECK_THROWS_AS(expr, Exc)                                             \
  do {                                                                         \
    ++::spt::checks();                                                         \
    bool caught = false;                                                       \
    try { (void)(expr); } catch (const Exc&) { caught = true; } catch (...) {} \
    if (!caught) ::spt::report_fail(__FILE__, __LINE__, "expected " #Exc " from " #expr); \
  } while (0)

// Iterate scypp result over golden::<name>_in, comparing to golden::<name>_out.
#define CHECK_ARR(name, call, rtol, atol)                                      \
  do {                                                                         \
    for (int _i = 0; _i < golden::name##_n; ++_i) {                            \
      double _g = (call)(golden::name##_in[_i]);                               \
      CHECK_CLOSE(_g, golden::name##_out[_i], rtol, atol);                     \
    }                                                                          \
  } while (0)

#define SPT_MAIN() int main() { return ::spt::run_all(); }
