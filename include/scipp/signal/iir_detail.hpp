#pragma once
// Shared internal IIR-design machinery (analog prototype → frequency transform →
// bilinear → transfer function), used by both design.cpp and design2.cpp.

#include <cmath>
#include <complex>
#include <vector>

#include "numpp/core/ndarray.hpp"
#include "scipp/error.hpp"
#include "scipp/linalg/detail.hpp"

namespace scipp::signal::iir {

using cd = std::complex<double>;
constexpr double kPi = 3.141592653589793238462643383279502884;
struct ZPK { std::vector<cd> z, p; double k; };

inline std::vector<cd> poly(const std::vector<cd>& roots) {
  std::vector<cd> c{1.0};
  for (cd r : roots) {
    std::vector<cd> nc(c.size() + 1, cd(0));
    for (size_t i = 0; i < c.size(); ++i) { nc[i] += c[i]; nc[i + 1] -= r * c[i]; }
    c = nc;
  }
  return c;
}
inline numpp::ndarray real_of(const std::vector<cd>& c) {
  std::vector<double> r(c.size());
  for (size_t i = 0; i < c.size(); ++i) r[i] = c[i].real();
  return scipp::linalg::detail::from_vec(r);
}
inline int degree(const ZPK& f) { return static_cast<int>(f.p.size() - f.z.size()); }

inline ZPK lp2lp(ZPK f, double wo) {
  int d = degree(f);
  for (cd& z : f.z) z *= wo;
  for (cd& p : f.p) p *= wo;
  f.k *= std::pow(wo, d);
  return f;
}
inline ZPK lp2hp(ZPK f, double wo) {
  int d = degree(f);
  cd nz = 1, dp = 1;
  for (cd z : f.z) nz *= -z;
  for (cd p : f.p) dp *= -p;
  for (cd& z : f.z) z = wo / z;
  for (cd& p : f.p) p = wo / p;
  for (int i = 0; i < d; ++i) f.z.push_back(0.0);
  f.k *= (nz / dp).real();
  return f;
}
inline ZPK lp2bp(ZPK f, double wo, double bw) {
  int d = degree(f);
  std::vector<cd> z, p;
  for (cd zz : f.z) { cd s = zz * bw / 2.0, r = std::sqrt(s * s - wo * wo); z.push_back(s + r); z.push_back(s - r); }
  for (cd pp : f.p) { cd s = pp * bw / 2.0, r = std::sqrt(s * s - wo * wo); p.push_back(s + r); p.push_back(s - r); }
  for (int i = 0; i < d; ++i) z.push_back(0.0);
  f.z = z; f.p = p; f.k *= std::pow(bw, d);
  return f;
}
inline ZPK lp2bs(ZPK f, double wo, double bw) {
  int d = degree(f);
  cd nz = 1, dp = 1;
  for (cd z : f.z) nz *= -z;
  for (cd p : f.p) dp *= -p;
  std::vector<cd> z, p;
  for (cd zz : f.z) { cd s = (bw / 2.0) / zz, r = std::sqrt(s * s - wo * wo); z.push_back(s + r); z.push_back(s - r); }
  for (cd pp : f.p) { cd s = (bw / 2.0) / pp, r = std::sqrt(s * s - wo * wo); p.push_back(s + r); p.push_back(s - r); }
  for (int i = 0; i < d; ++i) { z.push_back(cd(0, wo)); z.push_back(cd(0, -wo)); }
  f.z = z; f.p = p; f.k *= (nz / dp).real();
  return f;
}
inline ZPK bilinear(ZPK f, double fs) {
  int d = degree(f);
  double fs2 = 2.0 * fs;
  cd nz = 1, dp = 1;
  for (cd z : f.z) nz *= (fs2 - z);
  for (cd p : f.p) dp *= (fs2 - p);
  for (cd& z : f.z) z = (fs2 + z) / (fs2 - z);
  for (cd& p : f.p) p = (fs2 + p) / (fs2 - p);
  for (int i = 0; i < d; ++i) f.z.push_back(-1.0);
  f.k *= (nz / dp).real();
  return f;
}

struct BAcoef { numpp::ndarray b, a; };
inline BAcoef zpk2tf(const ZPK& f) {
  std::vector<cd> b = poly(f.z), a = poly(f.p);
  for (cd& c : b) c *= f.k;
  return {real_of(b), real_of(a)};
}

// Full design: prototype zpk → band transform (Wn pre-warped) → bilinear → (b,a).
inline BAcoef design(ZPK proto, const std::vector<double>& Wn, const std::string& btype) {
  double fs = 2.0;
  std::vector<double> warped;
  for (double w : Wn) warped.push_back(2.0 * fs * std::tan(kPi * w / fs));
  ZPK t;
  if (btype == "lowpass" || btype == "low") t = lp2lp(proto, warped[0]);
  else if (btype == "highpass" || btype == "high") t = lp2hp(proto, warped[0]);
  else {
    double bw = warped[1] - warped[0], wo = std::sqrt(warped[0] * warped[1]);
    if (btype == "bandpass" || btype == "band") t = lp2bp(proto, wo, bw);
    else if (btype == "bandstop" || btype == "stop") t = lp2bs(proto, wo, bw);
    else throw scipp::value_error("unknown btype " + btype);
  }
  return zpk2tf(bilinear(t, fs));
}

}  // namespace scipp::signal::iir
