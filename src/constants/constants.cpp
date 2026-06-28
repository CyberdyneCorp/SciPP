// scipp::constants implementation. The CODATA table is generated from the
// reference SciPy into codata_table.inc (frozen, committed) so lookups match the
// oracle exactly. Scale constants live in the header as constexpr.

#include "scipp/constants/constants.hpp"

#include <cctype>
#include <unordered_map>

#include "numpp/core/dtype.hpp"
#include "scipp/error.hpp"

namespace scipp::constants {
namespace {

const std::unordered_map<std::string, ConstantEntry>& table() {
  static const std::unordered_map<std::string, ConstantEntry> t = {
#include "codata_table.inc"
  };
  return t;
}

// Temperature scale -> Kelvin and back, keyed on the lowercased first letter.
char scale_key(std::string_view s) {
  if (s.empty()) throw scipp::value_error("empty temperature scale");
  return static_cast<char>(std::tolower(static_cast<unsigned char>(s[0])));
}

double to_kelvin(double v, char from) {
  switch (from) {
    case 'c': return v + 273.15;
    case 'k': return v;
    case 'f': return (v - 32.0) * 5.0 / 9.0 + 273.15;
    case 'r': return v * 5.0 / 9.0;
    default: throw scipp::value_error("unknown temperature scale");
  }
}

double from_kelvin(double k, char to) {
  switch (to) {
    case 'c': return k - 273.15;
    case 'k': return k;
    case 'f': return (k - 273.15) * 9.0 / 5.0 + 32.0;
    case 'r': return k * 9.0 / 5.0;
    default: throw scipp::value_error("unknown temperature scale");
  }
}

template <class F>
ndarray map(const ndarray& x, F&& f) {
  ndarray xc = x.astype(numpp::kFloat64).ascontiguousarray();
  ndarray out(xc.shape(), numpp::kFloat64);
  const double* in = xc.typed_data<double>();
  double* o = out.typed_data<double>();
  for (int64_t i = 0; i < xc.size(); ++i) o[i] = f(in[i]);
  return out;
}

}  // namespace

std::string_view codata_version() { return "CODATA (from reference SciPy)"; }

const ConstantEntry& physical_constant(const std::string& name) {
  auto it = table().find(name);
  if (it == table().end()) throw scipp::value_error("unknown physical constant: " + name);
  return it->second;
}

double value(const std::string& name) { return physical_constant(name).value; }
std::string unit(const std::string& name) { return physical_constant(name).unit; }
double precision(const std::string& name) {
  const ConstantEntry& e = physical_constant(name);
  return e.value != 0.0 ? e.uncertainty / e.value : 0.0;
}

double convert_temperature(double v, std::string_view from, std::string_view to) {
  return from_kelvin(to_kelvin(v, scale_key(from)), scale_key(to));
}

ndarray convert_temperature(const ndarray& v, std::string_view from, std::string_view to) {
  char f = scale_key(from), t = scale_key(to);
  return map(v, [f, t](double x) { return from_kelvin(to_kelvin(x, f), t); });
}

double lambda2nu(double lambda) { return c / lambda; }
double nu2lambda(double nu) { return c / nu; }
ndarray lambda2nu(const ndarray& lambda) { return map(lambda, [](double x) { return c / x; }); }
ndarray nu2lambda(const ndarray& nu) { return map(nu, [](double x) { return c / x; }); }

}  // namespace scipp::constants
