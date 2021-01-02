cmake_minimum_required(VERSION 3.10)

project(stan_pybind11)

find_package(Stan REQUIRED)

# Set C++ standard before finding pybind11
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(pybind11 REQUIRED CONFIG HINTS ${PYBIND11_DIR} ${PYBIND11_ROOT}
  $ENV{PYBIND11_DIR} $ENV{PYBIND11_ROOT})

# Create the binding library
pybind11_add_module(cpp SHARED
  stan/wrappers/dolfin.cpp
  stan/wrappers/common.cpp
  stan/wrappers/fem.cpp
  stan/wrappers/generation.cpp
  stan/wrappers/geometry.cpp
  stan/wrappers/graph.cpp
  stan/wrappers/io.cpp
  stan/wrappers/la.cpp
  stan/wrappers/log.cpp
  stan/wrappers/mesh.cpp
  stan/wrappers/nls.cpp
  stan/wrappers/refinement.cpp)

# Add srict compiler flags
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-Wall -Werror -pedantic" HAVE_PEDANTIC)
if (HAVE_PEDANTIC)
  target_compile_options(cpp PRIVATE -Wall;-Werror;-pedantic)
endif()

# Add Stan libraries and other config
target_link_libraries(cpp PRIVATE pybind11::module stan)

# Add to CMake search path
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
