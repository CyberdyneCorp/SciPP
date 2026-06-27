add_executable(scypp_tests
  main.cpp
  test_special.cpp
  test_constants.cpp)

target_link_libraries(scypp_tests PRIVATE scypp::scypp numpp::numpp)

target_include_directories(scypp_tests PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/golden)

add_test(NAME scypp_tests COMMAND scypp_tests)
