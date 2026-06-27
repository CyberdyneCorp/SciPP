// Peak analysis: find_peaks, peak_prominences, peak_widths.
#include "scypp/signal/signal.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

#include "numpp/core/dtype.hpp"
#include "scypp/linalg/detail.hpp"

namespace scypp::signal {
namespace {
namespace sd = scypp::linalg::detail;

std::vector<int> local_maxima(const std::vector<double>& x) {
  std::vector<int> mid;
  int n = static_cast<int>(x.size()), i = 1, imax = n - 1;
  while (i < imax) {
    if (x[i - 1] < x[i]) {
      int ahead = i + 1;
      while (ahead < imax && x[ahead] == x[i]) ++ahead;
      if (x[ahead] < x[i]) { mid.push_back((i + ahead - 1) / 2); i = ahead; }
    }
    ++i;
  }
  return mid;
}

std::vector<int> to_int(const ndarray& a) {
  std::vector<double> v = sd::to_vec(a);
  std::vector<int> r(v.size());
  for (size_t i = 0; i < v.size(); ++i) r[i] = static_cast<int>(std::llround(v[i]));
  return r;
}
ndarray int_vec(const std::vector<int>& v) {
  std::vector<double> d(v.begin(), v.end());
  return sd::from_vec(d);
}

void prominences(const std::vector<double>& x, const std::vector<int>& peaks,
                 std::vector<double>& prom, std::vector<int>& lb, std::vector<int>& rb) {
  int n = static_cast<int>(x.size());
  prom.resize(peaks.size()); lb.resize(peaks.size()); rb.resize(peaks.size());
  for (size_t k = 0; k < peaks.size(); ++k) {
    int peak = peaks[k];
    double lmin = x[peak]; int li = peak, i = peak;
    while (i >= 0 && x[i] <= x[peak]) { if (x[i] < lmin) { lmin = x[i]; li = i; } --i; }
    double rmin = x[peak]; int ri = peak; i = peak;
    while (i < n && x[i] <= x[peak]) { if (x[i] < rmin) { rmin = x[i]; ri = i; } ++i; }
    lb[k] = li; rb[k] = ri;
    prom[k] = x[peak] - std::max(lmin, rmin);
  }
}

void widths(const std::vector<double>& x, const std::vector<int>& peaks, double rel_height,
            const std::vector<double>& prom, const std::vector<int>& lb, const std::vector<int>& rb,
            std::vector<double>& w, std::vector<double>& wh, std::vector<double>& lip,
            std::vector<double>& rip) {
  w.resize(peaks.size()); wh.resize(peaks.size()); lip.resize(peaks.size()); rip.resize(peaks.size());
  for (size_t k = 0; k < peaks.size(); ++k) {
    int peak = peaks[k];
    double height = x[peak] - prom[k] * rel_height;
    wh[k] = height;
    int i = peak;
    while (lb[k] < i && height < x[i]) --i;
    double left = i;
    if (x[i] < height) left += (height - x[i]) / (x[i + 1] - x[i]);
    i = peak;
    while (i < rb[k] && height < x[i]) ++i;
    double right = i;
    if (x[i] < height) right -= (height - x[i]) / (x[i - 1] - x[i]);
    lip[k] = left; rip[k] = right; w[k] = right - left;
  }
}
}  // namespace

ProminencesResult peak_prominences(const ndarray& x, const ndarray& peaks) {
  std::vector<double> xv = sd::to_vec(x);
  std::vector<int> pk = to_int(peaks), lb, rb;
  std::vector<double> prom;
  prominences(xv, pk, prom, lb, rb);
  return {sd::from_vec(prom), int_vec(lb), int_vec(rb)};
}

WidthsResult peak_widths(const ndarray& x, const ndarray& peaks, double rel_height) {
  std::vector<double> xv = sd::to_vec(x);
  std::vector<int> pk = to_int(peaks), lb, rb;
  std::vector<double> prom, w, wh, lip, rip;
  prominences(xv, pk, prom, lb, rb);
  widths(xv, pk, rel_height, prom, lb, rb, w, wh, lip, rip);
  return {sd::from_vec(w), sd::from_vec(wh), sd::from_vec(lip), sd::from_vec(rip)};
}

FindPeaksResult find_peaks(const ndarray& x, std::optional<double> height,
                           std::optional<int64_t> distance, std::optional<double> prominence,
                           std::optional<double> width) {
  std::vector<double> xv = sd::to_vec(x);
  std::vector<int> peaks = local_maxima(xv);

  if (height) {
    std::vector<int> keep;
    for (int p : peaks) if (xv[p] >= *height) keep.push_back(p);
    peaks = keep;
  }
  if (distance) {
    int d = static_cast<int>(*distance);
    std::vector<int> idx(peaks.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int a, int b) { return xv[peaks[a]] < xv[peaks[b]]; });
    std::vector<char> keep(peaks.size(), 1);
    for (auto it = idx.rbegin(); it != idx.rend(); ++it) {
      int i = *it;
      if (!keep[i]) continue;
      for (int j = i - 1; j >= 0 && peaks[i] - peaks[j] < d; --j) keep[j] = 0;
      for (int j = i + 1; j < static_cast<int>(peaks.size()) && peaks[j] - peaks[i] < d; ++j) keep[j] = 0;
    }
    std::vector<int> kept;
    for (size_t i = 0; i < peaks.size(); ++i) if (keep[i]) kept.push_back(peaks[i]);
    peaks = kept;
  }

  std::vector<double> prom, w, wh, lip, rip;
  std::vector<int> lb, rb;
  prominences(xv, peaks, prom, lb, rb);
  if (prominence) {
    std::vector<int> kp; std::vector<double> kprom; std::vector<int> klb, krb;
    for (size_t i = 0; i < peaks.size(); ++i)
      if (prom[i] >= *prominence) { kp.push_back(peaks[i]); kprom.push_back(prom[i]); klb.push_back(lb[i]); krb.push_back(rb[i]); }
    peaks = kp; prom = kprom; lb = klb; rb = krb;
  }
  widths(xv, peaks, 0.5, prom, lb, rb, w, wh, lip, rip);
  if (width) {
    std::vector<int> kp; std::vector<double> kprom, kw;
    for (size_t i = 0; i < peaks.size(); ++i)
      if (w[i] >= *width) { kp.push_back(peaks[i]); kprom.push_back(prom[i]); kw.push_back(w[i]); }
    peaks = kp; prom = kprom; w = kw;
  }
  return {int_vec(peaks), sd::from_vec(prom), sd::from_vec(w), int_vec(lb), int_vec(rb)};
}

}  // namespace scypp::signal
