cmake_minimum_required(VERSION 3.8)
project(caret_analyze_cpp_impl)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)

add_subdirectory(pybind11)
include_directories(
  include
  pybind11/include
  json/include
  indicators/include
)
pybind11_add_module(record_cpp_impl
src/pybind.cpp
src/record.cpp
src/records_base.cpp
src/records_vector_impl.cpp
src/records_map_impl.cpp
src/iterator_base.cpp
src/iterator_vector_impl.cpp
src/iterator_map_impl.cpp
src/progress.cpp
src/column_manager.cpp
)

install(
  TARGETS record_cpp_impl
  EXPORT export_record_cpp_impl
  LIBRARY DESTINATION lib/python3.8/site-packages
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)


add_executable(merge_sample
src/merge_sample.cpp
src/record.cpp
src/records_base.cpp
src/records_vector_impl.cpp
src/records_map_impl.cpp
src/iterator_base.cpp
src/iterator_vector_impl.cpp
src/iterator_map_impl.cpp
src/progress.cpp
src/column_manager.cpp
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
