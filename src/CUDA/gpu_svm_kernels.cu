#include "gpu_svm_kernels.h"

template<typename T>
__global__
void batchPredict_kernel(T *weight, T *data, T *result, int instances, int dim){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx < 0 || idx > instances){
        return;
    }
    result[idx] = innerProduct(weight, data[idx*dim], dim);
}

template<typename T>
__global__
void batchLearn_kernel(T *weight, T *data, T *result, int *labels, int batchSize, T eta, T lambda){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx < 0 || idx > batchSize){
        return;
    }
}