cmake_minimum_required(VERSION 3.18)
project(cel4o LANGUAGES CXX)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

add_library(cel4o_core STATIC src/common.cpp src/cel4o_d.cpp src/cel4o_q.cpp)
target_include_directories(cel4o_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(cel4o_core PUBLIC cxx_std_20)
set_target_properties(cel4o_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

pybind11_add_module(_cel4o src/python_bindings.cpp)
target_link_libraries(_cel4o PRIVATE cel4o_core)
target_compile_features(_cel4o PRIVATE cxx_std_20)

install(TARGETS _cel4o DESTINATION compact_matrix_sqrt)
