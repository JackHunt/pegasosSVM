#ifndef PEGASOS_GPU_SVM_KERNELS_HEADER
#define PEGASOS_GPU_SVM_KERNELS_HEADER

#include "../shared/general.h"
#include "../shared/training.h"

template<typename T>
__global__
void batchPredict_kernel(T *weight, T *data, T *result, int instances, int dim);

template<typename T>
__global__
void batchLearn_kernel(T *weight, T *data, T *result, int *labels, int batchSize, T eta, T lambda);

#endif