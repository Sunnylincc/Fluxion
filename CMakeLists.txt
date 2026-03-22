cmake_minimum_required(VERSION 3.20)
project(fluxion_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(pybind11 REQUIRED)

pybind11_add_module(fluxion_cpp
  cpp/bindings/pybind_module.cpp
  cpp/kv_cache/block_allocator.cpp
)

target_include_directories(fluxion_cpp PRIVATE cpp)
