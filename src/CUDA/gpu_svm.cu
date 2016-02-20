#include "gpu_svm.h"

//------------------------------------------------------------------------------
//Public members.
//------------------------------------------------------------------------------
template<typename T>
gpuSVM<T>::gpuSVM(int D, T lambda) : dataDimension(D), lambda(lambda){
    //
}

template<typename T>
gpuSVM::~gpuSVM(){
    //
}

template<typename T>
void gpuSVM<T>::train(T *data, int *labels, int instances, int batchSize){
    //
}

template<typename T>
T gpuSVM<T>::predict(T *data){
    //
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