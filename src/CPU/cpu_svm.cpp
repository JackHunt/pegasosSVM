#include "cpu_svm.h"

using namespace pegasos;

//------------------------------------------------------------------------------
//Public Constructor and Destructor.
//------------------------------------------------------------------------------
template<typename T>
cpuSVM<T>::cpuSVM(int D, T lambda) : dataDimension(D){
    this->weights = new T[D];
    for(int i=0; i<D; i++){
        this->weights[i] = (T)0.0;
    }
    
    this->eta = (T)0.0;
}

template<typename T>
cpuSVM<T>::~cpuSVM(){
    delete[] weights;
}

//------------------------------------------------------------------------------
//Public members.
//------------------------------------------------------------------------------
template<typename T>
void cpuSVM<T>::train(T* data, int *labels, int instances, int batchSize){
    std::vector<int> batchIndices = getBatch(batchSize, instances);
    
    //Do OpenMP parallel reduction.
}

template<typename T>
T cpuSVM<T>::predict(T data){
    //
}

template<typename T>
void cpuSVM<T>::predict(T* data, T* result, int instances){
    //
}

//------------------------------------------------------------------------------
//Private members.
//------------------------------------------------------------------------------
template<typename T>
std::vector<int> cpuSVM<T>::getBatch(int batchSize, int numElements){
    if(batchSize < numElements){
        std::vector<int> batchIndices;
        for(int i=0; i<batchSize; i++){
            batchIndices.push_back((rand() % batchSize));
        }
        return batchIndices;
    }else{
        std::vector<int> batchIndices(numElements);
        
        #pragma omp parallel for
        for(int i=0; i<numElements; i++){
            batchIndices[i] = i;
        }
        return batchIndices;
    }
}