#ifndef TRAINING_SHARED_HEADER
#define	TRAINING_SHARED_HEADER

#include "shared.h"

template<typename T>
__SHARED_CODE__
inline T computeEta(T *lambda, int t){
    return (T)(1.0/(lambda*(T)t));
}

template<typename T>
__SHARED_CODE__
inline void weightUpdate(T *weights, T eta, T lambda, int batchSize, T *batchSum, int D){
    T c1 = (T)1.0-(eta*lambda);
    T c2 = eta/(T)batchSize;
    for(int i=0; i<D; i++){
        weights[i] = c1*weights[i] + c2*batchSum[i];
    }
}

#endif

