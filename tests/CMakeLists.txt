add_executable(scypp_tests
  main.cpp
  test_special.cpp
  test_special_airy_elliptic.cpp
  test_special_spherical_sici.cpp
  test_special_erf_extras.cpp
  test_special_misc.cpp
  test_special_hypergeometric.cpp
  test_foundation.cpp
  test_constants.cpp
  test_linalg.cpp
  test_fft.cpp
  test_optimize.cpp
  test_integrate.cpp
  test_interpolate.cpp
  test_stats.cpp
  test_signal.cpp
  test_signal_extras.cpp
  test_signal_discrete.cpp
  test_sparse.cpp
  test_sparse_csgraph.cpp
  test_spatial.cpp
  test_ndimage.cpp
  test_cluster_io.cpp
  test_stats_discrete.cpp
  test_optimize_lp.cpp
  test_interpolate_griddata.cpp
  test_interpolate_bspline.cpp
  test_integrate_radau.cpp
  test_integrate_bdf.cpp
  test_integrate_nested.cpp
  test_integrate_bvp.cpp
  test_stats_normality.cpp
  test_odr.cpp)

target_link_libraries(scypp_tests PRIVATE scypp::scypp numpp::numpp)

target_include_directories(scypp_tests PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/golden)

target_compile_definitions(scypp_tests PRIVATE SCYPP_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/golden")

add_test(NAME scypp_tests COMMAND scypp_tests)
