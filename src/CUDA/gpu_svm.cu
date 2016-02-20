#include "gpu_svm.h"

//------------------------------------------------------------------------------
//Public members.
//------------------------------------------------------------------------------
template<typename T>
gpuSVM<T>::gpuSVM(int D, T lambda) : dataDimension(D), lambda(lambda){
    CUDA_CHECK(cudaMalloc((void**)&this->weights, D*sizeof(T)));
    this->eta = (T)0.0;
}

template<typename T>
gpuSVM::~gpuSVM(){
    CUDA_CHECK(this->weights);
}

template<typename T>
void gpuSVM<T>::train(T *data, int *labels, int instances, int batchSize){
    //
}

template<typename T>
T gpuSVM<T>::predict(T *data){
    return innerProduct(weights, data);
}

template<typename T>
void gpuSVM<T>::predict(T* data, T* result, int instances){
    //
}

//------------------------------------------------------------------------------
//Protected members.
//------------------------------------------------------------------------------
template<typename T>
thrust::vector<int> gpuSVM<T>::getBatch(int batchSize, int numElements){
    if(batchSize < numElements){
        //
    }else{
        //
    }
}