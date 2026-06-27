add_executable(scypp_tests
  main.cpp
  test_special.cpp
  test_constants.cpp
  test_linalg.cpp
  test_fft.cpp
  test_optimize.cpp
  test_integrate.cpp
  test_interpolate.cpp
  test_stats.cpp
  test_signal.cpp
  test_signal_extras.cpp
  test_sparse.cpp)

target_link_libraries(scypp_tests PRIVATE scypp::scypp numpp::numpp)

target_include_directories(scypp_tests PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/golden)

add_test(NAME scypp_tests COMMAND scypp_tests)
