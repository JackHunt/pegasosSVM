#TO-DO: Check actually required version.
cmake_minimum_required(VERSION 2.8)
project(pegasosSVM)

#Set compiler flags for Windows and Unix, respectively.
if(MSVC_IDE)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
  add_definitions(-DUSING_CMAKE=1)
else(MSVC_IDE)
  set(CMAKE_CXX_FLAGS_DEBUG "-Wall -fPIC -g -march=native")
  set(CMAKE_CXX_FLAGS_RELEASE "-Wall -fPIC -O3 -march=native")
endif(MSVC_IDE)

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

#Add project sources
set(PEGASOS_SHARED_HEADERS
  src/shared/shared.h
  src/shared/svm.h
  src/shared/general.h
  src/shared/training.h
)

set(PEGASOS_CPU_FILES
  src/CPU/cpu_svm.h
  src/CPU/cpu_svm.cpp
)

set(PEGASOS_GPU_FILES
  src/CUDA/gpu_svm.h
  src/CUDA/gpu_svm.cu
)

#Build libs.
if(WITH_CUDA)
  list(APPEND CUDA_NVCC_FLAGS "-arch=sm_20; -O2;-DVERBOSE")
  cuda_add_library(SVM SHARED ${PEGASOS_SHARED_HEADERS} 
    ${PEGASOS_CPU_FILES} ${PEGASOS_GPU_FILES})
else(WITH_CUDA)
  add_definitions(-DCOMPILE_WITHOUT_CUDA)
  add_library(SVM SHARED ${PEGASOS_SHARED_HEADERS} ${PEGASOS_CPU_FILES})
endif(WITH_CUDA)
