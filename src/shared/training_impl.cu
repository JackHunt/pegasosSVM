#include "general.h"

template<typename T>
__host__ __device__ 
T computeEtaDevice(T *lambda, int t){
    return (T)(1.0/(lambda*(T)t));
}

template<typename T>
T computeEta(T *lambda, int t){
    return computeEtaDevice(lambda, t);
}

template<typename T>
__host__ __device__ 
void weightUpdateDevice(T *weights, T eta, T lambda, int batchSize, T *batchSum, int D){
    T c1 = (T)1.0-(eta*lambda);
    T c2 = eta/(T)batchSize;
    for(int i=0; i<D; i++){
        weights[i] = c1*weights[i] + c2*batchSum[i];
    }
}

template<typename T>
void weightUpdate(T *weights, T eta, T lambda, int batchSize, T *batchSum, int D){
    weightUpdateDevice(weights, eta, lambda, batchSize, batchSum, D);
}