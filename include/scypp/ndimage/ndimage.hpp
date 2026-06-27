#pragma once
// scypp::ndimage — port of scipy.ndimage (Phase 11): filters, morphology,
// measurements, and geometric transforms over 2-D numpp::ndarray images.

#include <cstdint>
#include <string>
#include <vector>

#include "numpp/core/ndarray.hpp"

namespace scypp::ndimage {

using numpp::ndarray;
enum class Backend { Cpu, Device };
Backend last_backend();

// ---- filters ----
ndarray correlate1d(const ndarray& input, const ndarray& weights, int axis = -1,
                    const std::string& mode = "reflect", double cval = 0.0, int origin = 0);
ndarray convolve1d(const ndarray& input, const ndarray& weights, int axis = -1,
                   const std::string& mode = "reflect", double cval = 0.0, int origin = 0);
ndarray correlate(const ndarray& input, const ndarray& weights, const std::string& mode = "reflect",
                  double cval = 0.0);
ndarray convolve(const ndarray& input, const ndarray& weights, const std::string& mode = "reflect",
                 double cval = 0.0);
ndarray uniform_filter1d(const ndarray& input, int size, int axis = -1,
                         const std::string& mode = "reflect", double cval = 0.0);
ndarray uniform_filter(const ndarray& input, int size, const std::string& mode = "reflect",
                       double cval = 0.0, Backend forced = Backend::Cpu);
ndarray gaussian_filter1d(const ndarray& input, double sigma, int axis = -1, double truncate = 4.0,
                          const std::string& mode = "reflect", double cval = 0.0);
ndarray gaussian_filter(const ndarray& input, double sigma, double truncate = 4.0,
                        const std::string& mode = "reflect", double cval = 0.0,
                        Backend forced = Backend::Cpu);
ndarray median_filter(const ndarray& input, int size, const std::string& mode = "reflect", double cval = 0.0);
ndarray minimum_filter(const ndarray& input, int size, const std::string& mode = "reflect", double cval = 0.0);
ndarray maximum_filter(const ndarray& input, int size, const std::string& mode = "reflect", double cval = 0.0);
ndarray sobel(const ndarray& input, int axis = -1, const std::string& mode = "reflect", double cval = 0.0);
ndarray prewitt(const ndarray& input, int axis = -1, const std::string& mode = "reflect", double cval = 0.0);
ndarray laplace(const ndarray& input, const std::string& mode = "reflect", double cval = 0.0);

// ---- morphology ----
ndarray binary_erosion(const ndarray& input, int iterations = 1);
ndarray binary_dilation(const ndarray& input, int iterations = 1);
ndarray binary_opening(const ndarray& input, int iterations = 1);
ndarray binary_closing(const ndarray& input, int iterations = 1);
ndarray grey_erosion(const ndarray& input, int size);
ndarray grey_dilation(const ndarray& input, int size);
ndarray distance_transform_edt(const ndarray& input);

// ---- measurements ----
struct LabelResult { ndarray labels; int num_features; };
LabelResult label(const ndarray& input);
ndarray center_of_mass(const ndarray& input);
double sum_labels(const ndarray& input);
double mean(const ndarray& input);
double maximum(const ndarray& input);
double minimum(const ndarray& input);
ndarray sum_labels(const ndarray& input, const ndarray& labels, const std::vector<int>& index);
ndarray mean(const ndarray& input, const ndarray& labels, const std::vector<int>& index);

// ---- geometric transforms ----
ndarray map_coordinates(const ndarray& input, const ndarray& coordinates, int order = 1,
                        const std::string& mode = "constant", double cval = 0.0);
ndarray affine_transform(const ndarray& input, const ndarray& matrix, const ndarray& offset,
                         int order = 1, const std::string& mode = "constant", double cval = 0.0);
ndarray shift(const ndarray& input, const std::vector<double>& shift, int order = 1,
              const std::string& mode = "constant", double cval = 0.0);
ndarray zoom(const ndarray& input, double factor, int order = 1, const std::string& mode = "constant",
             double cval = 0.0);
ndarray rotate(const ndarray& input, double angle, int order = 1, const std::string& mode = "constant",
               double cval = 0.0);

}  // namespace scypp::ndimage
