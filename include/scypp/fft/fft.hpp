#pragma once
// scypp::fft — port of scipy.fft (Phase 3 subset). The complex/real FFTs delegate
// to NumPP (identical norm conventions); the DCT/DST family and next_fast_len are
// implemented here. scypp::fftpack re-exports the legacy entry points.

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "numpp/core/ndarray.hpp"

namespace scypp::fft {

using numpp::ndarray;

// ---- discrete Fourier transforms (delegate to NumPP) ----
ndarray fft(const ndarray& a, std::optional<int64_t> n = std::nullopt, int64_t axis = -1,
            const std::string& norm = "backward");
ndarray ifft(const ndarray& a, std::optional<int64_t> n = std::nullopt, int64_t axis = -1,
             const std::string& norm = "backward");
ndarray rfft(const ndarray& a, std::optional<int64_t> n = std::nullopt, int64_t axis = -1,
             const std::string& norm = "backward");
ndarray irfft(const ndarray& a, std::optional<int64_t> n = std::nullopt, int64_t axis = -1,
              const std::string& norm = "backward");
ndarray hfft(const ndarray& a, std::optional<int64_t> n = std::nullopt, int64_t axis = -1,
             const std::string& norm = "backward");
ndarray ihfft(const ndarray& a, std::optional<int64_t> n = std::nullopt, int64_t axis = -1,
              const std::string& norm = "backward");

ndarray fft2(const ndarray& a, std::optional<std::vector<int64_t>> s = std::nullopt,
             const std::string& norm = "backward");
ndarray ifft2(const ndarray& a, std::optional<std::vector<int64_t>> s = std::nullopt,
              const std::string& norm = "backward");
ndarray fftn(const ndarray& a, std::optional<std::vector<int64_t>> s = std::nullopt,
             const std::string& norm = "backward");
ndarray ifftn(const ndarray& a, std::optional<std::vector<int64_t>> s = std::nullopt,
              const std::string& norm = "backward");
ndarray rfftn(const ndarray& a, std::optional<std::vector<int64_t>> s = std::nullopt,
              const std::string& norm = "backward");
ndarray irfftn(const ndarray& a, std::optional<std::vector<int64_t>> s = std::nullopt,
               const std::string& norm = "backward");

// ---- discrete cosine / sine transforms (types 1-4) ----
// norm in {"backward" (default), "ortho", "forward"} matches scipy.fft. The
// "ortho" variant is orthogonalized so the coefficient matrix is orthonormal.
ndarray dct(const ndarray& a, int type = 2, int64_t axis = -1, const std::string& norm = "backward");
ndarray idct(const ndarray& a, int type = 2, int64_t axis = -1, const std::string& norm = "backward");
ndarray dst(const ndarray& a, int type = 2, int64_t axis = -1, const std::string& norm = "backward");
ndarray idst(const ndarray& a, int type = 2, int64_t axis = -1, const std::string& norm = "backward");

// N-D variants: apply the 1-D transform over `axes` (default: all axes).
ndarray dctn(const ndarray& a, int type = 2, std::optional<std::vector<int64_t>> axes = std::nullopt,
             const std::string& norm = "backward");
ndarray idctn(const ndarray& a, int type = 2, std::optional<std::vector<int64_t>> axes = std::nullopt,
              const std::string& norm = "backward");
ndarray dstn(const ndarray& a, int type = 2, std::optional<std::vector<int64_t>> axes = std::nullopt,
             const std::string& norm = "backward");
ndarray idstn(const ndarray& a, int type = 2, std::optional<std::vector<int64_t>> axes = std::nullopt,
              const std::string& norm = "backward");

// ---- helpers ----
ndarray fftfreq(int64_t n, double d = 1.0);
ndarray rfftfreq(int64_t n, double d = 1.0);
ndarray fftshift(const ndarray& a, std::optional<std::vector<int64_t>> axes = std::nullopt);
ndarray ifftshift(const ndarray& a, std::optional<std::vector<int64_t>> axes = std::nullopt);
int64_t next_fast_len(int64_t n, bool real = false);

}  // namespace scypp::fft

namespace scypp::fftpack {
// Legacy API surface — re-exports of scypp::fft.
using numpp::ndarray;
ndarray fft(const ndarray& a, std::optional<int64_t> n = std::nullopt, int64_t axis = -1);
ndarray ifft(const ndarray& a, std::optional<int64_t> n = std::nullopt, int64_t axis = -1);
ndarray rfft(const ndarray& a, std::optional<int64_t> n = std::nullopt, int64_t axis = -1);
ndarray irfft(const ndarray& a, std::optional<int64_t> n = std::nullopt, int64_t axis = -1);
ndarray dct(const ndarray& a, int type = 2, int64_t axis = -1);
ndarray idct(const ndarray& a, int type = 2, int64_t axis = -1);
ndarray dst(const ndarray& a, int type = 2, int64_t axis = -1);
ndarray idst(const ndarray& a, int type = 2, int64_t axis = -1);
}  // namespace scypp::fftpack
