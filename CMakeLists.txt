cmake_minimum_required(VERSION 3.10.0)
project(parallel VERSION 0.1.0 LANGUAGES C CXX)

include(OpenMP.cmake)

add_executable(default default.cpp)
target_link_libraries(default PRIVATE OpenMP::OpenMP_CXX)

add_executable(explicit explicit_scheduling.cpp)
target_link_libraries(explicit PRIVATE OpenMP::OpenMP_CXX)