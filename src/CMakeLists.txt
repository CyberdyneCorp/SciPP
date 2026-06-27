add_library(scypp
  special/gamma.cpp
  special/erf.cpp
  special/expint.cpp
  special/bessel.cpp
  special/orthogonal.cpp
  special/combinatorics.cpp
  special/reductions.cpp
  constants/constants.cpp
  linalg/basic.cpp
  linalg/decomp.cpp
  linalg/eigen.cpp
  linalg/matfuncs.cpp
  linalg/special.cpp
  fft/fft.cpp
  fft/realtransforms.cpp
  optimize/roots_scalar.cpp
  optimize/minimize_scalar.cpp
  optimize/minimize.cpp
  optimize/least_squares.cpp)

add_library(scypp::scypp ALIAS scypp)

target_compile_features(scypp PUBLIC cxx_std_20)

target_include_directories(scypp PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/generated>
  $<INSTALL_INTERFACE:include>)

target_link_libraries(scypp PUBLIC numpp::numpp)
