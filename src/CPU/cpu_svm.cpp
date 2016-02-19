#include "cpu_svm.h"

using namespace pegasos;

//------------------------------------------------------------------------------
//Public Constructor and Destructor.
//------------------------------------------------------------------------------
template<typename T>
cpuSVM::cpuSVM(int D, T lambda) : dataDimension(D){
    this->weights = new T[D];
    
    #pragma omp parallel for
    for(int i=0; i<D; i++){
        this->weights[i] = (T)0.0;
    }
    
    this->eta = (T)0.0;
}

template<typename T>
cpuSVM::~cpuSVM(){
    delete[] weights;
}

//------------------------------------------------------------------------------
//Public members.
//------------------------------------------------------------------------------
template<typename T>
void cpuSVM::train(T* data, int *labels, int instances, int batchSize){
    int numBatches = ceil(instances/batchSize);
}

template<typename T>
T cpuSVM::predict(T data){
    //
}

template<typename T>
void cpuSVM::predict(T* data, T* result, int instances){
    //
}

//------------------------------------------------------------------------------
//Protected members.
//------------------------------------------------------------------------------
template<typename T>
void doGemm(){
    //
}