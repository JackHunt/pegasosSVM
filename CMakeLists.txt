#TO-DO: Check actually required version.
cmake_minimum_required(VERSION 2.6)
project(pegasosSVM)

#Set compiler flags for Windows and Unix, respectively.
if(MSVC_IDE)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
  add_definitions(-DUSING_CMAKE=1)
else(MSVC_IDE)
  set(CMAKE_CXX_FLAGS_DEBUG "-Wall -fPIC -g -march=native")
  set(CMAKE_CXX_FLAGS_RELEASE "-Wall -fPIC -O3 -march=native")
endif(MSVC_IDE)

#Require BLAS as we at least always want a CPU implementation.
find_package(BLAS REQUIRED)

#Attempt to find CUDA.
find_package(CUDA QUIET)
option(WITH_CUDA " To be built with CUDA support?" ${CUDA_FOUND})
if(WITH_CUDA)
  include_directories(${CUDA_INCLUDE_DIRS})
else()
  add_definitions(-DCOMPILE_WITHOUT_CUDA)
endif()

#Determine if we can use OpenMP and append to compiler flags accordingly.
option(USE_OpenMP "Use OpenMP" ON)
if(USE_OpenMP)
  find_package(OpenMP)
  if(OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  ENDIF()
endif()

#Add project directories.
add_subdirectory(src)