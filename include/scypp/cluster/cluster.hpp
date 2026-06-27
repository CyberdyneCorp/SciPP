#pragma once
// scypp::cluster — port of scipy.cluster (Phase 12): vector quantization and
// hierarchical clustering.

#include <string>

#include "numpp/core/ndarray.hpp"

namespace scypp::cluster {

using numpp::ndarray;

// ---- vq ----
ndarray whiten(const ndarray& obs);
struct VQResult { ndarray code, dist; };
VQResult vq(const ndarray& obs, const ndarray& code_book);
struct KMeans2Result { ndarray centroids, labels; };
KMeans2Result kmeans2(const ndarray& data, const ndarray& init, int iter = 10);

// ---- hierarchy ----
ndarray linkage(const ndarray& y, const std::string& method = "single",
                const std::string& metric = "euclidean");
ndarray fcluster(const ndarray& Z, double t, const std::string& criterion = "distance");
ndarray cophenet(const ndarray& Z);

}  // namespace scypp::cluster
