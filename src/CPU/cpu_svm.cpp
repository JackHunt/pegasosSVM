#include "cpu_svm.h"

using namespace pegasos;

//------------------------------------------------------------------------------
//Public members.
//------------------------------------------------------------------------------
template<typename T>
cpuSVM<T>::cpuSVM(int D, T lambda) : dataDimension(D), lambda(lambda){
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

template<typename T>
void cpuSVM<T>::train(T *data, int *labels, int instances, int batchSize){
    std::vector<int> batch = getBatch(batchSize, instances);
    DVector<T> *batchSum;
    #if defined(_OPENMP)
        #pragma omp parallel reduction(+:batchSum)
    #endif
    for(std::vector<int>::iterator i=batch.begin(); i!=batch.end(); ++i){
        T inner = innerProduct(weights, data[*i]);
        if(labels[*i]*inner < 1.0){
            DVector<T> dataVec(dataDimension, data[*i*dataDimension]);
            dataVec *= (T)labels[*i];
            batchSum += dataVec;
        }
    }
    weightUpdate(weights, eta, lambda, batchSize, batchSum, dataDimension);
}

template<typename T>
T cpuSVM<T>::predict(T *data){
    return innerProduct(weights, data);
}

template<typename T>
void cpuSVM<T>::predict(T* data, T* result, int instances){
    #if defined(_OPENMP)
        #pragma omp parallel for
    #endif
    for(int i=0; i<instances; i++){
        result[i] = innerProduct(weights, data[i*dataDimension]);
    }
}

//------------------------------------------------------------------------------
//Protected members.
//------------------------------------------------------------------------------
template<typename T>
std::vector<int> cpuSVM<T>::getBatch(int batchSize, int numElements){
    if(batchSize < numElements){
        std::vector<int> batchIndices(batchSize);
        #if defined(_OPENMP)
            #pragma omp parallel for
        #endif
        for(int i=0; i<batchSize; i++){
            batchIndices[i] = ((rand() % batchSize));
        }
        return batchIndices;
    }else{
        std::vector<int> batchIndices(numElements);
        #if defined(_OPENMP)
            #pragma omp parallel for
        #endif
        for(int i=0; i<numElements; i++){
            batchIndices[i] = i;
        }
        return batchIndices;
    }
}