#pragma once
// scypp::io — port of scipy.io (Phase 12): Matrix Market, WAV, and ARFF.

#include <string>

#include "numpp/core/ndarray.hpp"

namespace scypp::io {

using numpp::ndarray;

// ---- Matrix Market ----
ndarray mmread(const std::string& path);                 // returns a dense matrix
void mmwrite(const std::string& path, const ndarray& a);

// ---- WAV (integer PCM) ----
struct WavData { int rate; ndarray data; };
WavData wavread(const std::string& path);
void wavwrite(const std::string& path, int rate, const ndarray& data);

// ---- ARFF (numeric attributes) ----
ndarray loadarff(const std::string& path);

}  // namespace scypp::io
