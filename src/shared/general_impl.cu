#include "general.h"

template<typename T>
__host__ __device__ T computeEtaDevice(T *lambda, int t){
    return (T)(1.0/(lambda*(T)t));
}

template<typename T>
__host__ __device__ T innerProductDevice(T *w, T *x, int dim){
    T tmp = (T)0.0;
    for(int i=0; i<dim; i++){
        tmp += w[i] * x[i];
    }
    return tmp;
}

template<typename T>
T computeEta(T *lambda, int t){
    return computeEtaDevice(lambda, t);
}

template<typename T>
T innerProduct(T *w, T *x, int dim){
    return innerProductDevice(w, x, dim);
}