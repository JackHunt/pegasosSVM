set(PEGASOS_SHARED_HEADERS
  shared/svm.h
  shared/general.h
)

set(PEGASOS_CPU_FILES
  CPU/cpu_svm.h
  CPU/cpu_svm.cpp
)

set(PEGASOS_GPU_FILES
  CUDA/cuda_svm.h
  CUDA/cuda_svm.cu
  shared/general_impl.cu
)

if(WITH_CUDA)
  cuda_add_library(SVM ${PEGASOS_SHARED_HEADERS} 
    ${PEGASOS_CPU_FILES} ${PEGASOS_GPU_FILES})
else(WITH_CUDA)
  add_definitions(-DCOMPILE_WITHOUT_CUDA)
  add_library(SVM ${PEGASOS_SHARED_HEADERS} ${PEGASOS_CPU_FILES})
endif(WITH_CUDA)