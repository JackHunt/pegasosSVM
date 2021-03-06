#TO-DO: Check actually required version.
cmake_minimum_required(VERSION 2.8)
project(pegasosSVM)

#Set compiler flags for Windows and Unix, respectively.
if(MSVC_IDE)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
  add_definitions(-DUSING_CMAKE=1)
else(MSVC_IDE)
  set(CMAKE_CXX_FLAGS_DEBUG "-Wall -fPIC -g -march=native -fpermissive")
  set(CMAKE_CXX_FLAGS_RELEASE "-Wall -fPIC -O3 -march=native -fpermissive")
endif(MSVC_IDE)

#Attempt to find CBLAS.
find_package(BLAS REQUIRED)
include_directories(${BLAS_INCLUDE_DIRS})
find_package(LAPACK REQUIRED)
include_directories(${LAPACK_INCLUDE_DIRS})

#Attempt to find CUDA.
find_package(CUDA QUIET)
option(WITH_CUDA " To be built with CUDA support?" ${CUDA_FOUND})
if(WITH_CUDA)
  include_directories(${CUDA_INCLUDE_DIRS})
else(WITH_CUDA)
  add_definitions(-DCOMPILE_WITHOUT_CUDA)
endif(WITH_CUDA)

#Determine if we can use OpenMP and append to compiler flags accordingly.
option(USE_OpenMP "Use OpenMP" ON)
if(USE_OpenMP)
  find_package(OpenMP)
  if(OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  endif(OPENMP_FOUND)
endif(USE_OpenMP)

#Add project sources
set(PEGASOS_SHARED_HEADERS
  ${CMAKE_CURRENT_SOURCE_DIR}/src/shared/shared.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/shared/svm.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/shared/general.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/shared/training.hpp
)

set(PEGASOS_CPU_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/src/CPU/cpu_svm.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/CPU/cpu_svm.cpp
)

set(PEGASOS_GPU_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/src/CUDA/gpu_svm.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/CUDA/gpu_svm_kernels.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/CUDA/gpu_svm.cu
)

#Build libs.
if(WITH_CUDA)
  list(APPEND CUDA_NVCC_FLAGS "-arch=sm_30; -O2;-DVERBOSE")
  cuda_add_library(SVM SHARED ${PEGASOS_SHARED_HEADERS} ${PEGASOS_CPU_FILES} ${PEGASOS_GPU_FILES})
  cuda_add_cublas_to_target(SVM)
else(WITH_CUDA)
  add_definitions(-DCOMPILE_WITHOUT_CUDA)
  add_library(SVM SHARED ${PEGASOS_SHARED_HEADERS} ${PEGASOS_CPU_FILES})
  target_link_libraries(${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
endif(WITH_CUDA)
