#include "general.h"

template<typename T>
__host__ __device__ 
T innerProductDevice(T *w, T *x, int dim){
    T tmp = (T)0.0;
    for(int i=0; i<dim; i++){
        tmp += w[i] * x[i];
    }
    return tmp;
}

template<typename T>
T innerProduct(T *w, T *x, int dim){
    return innerProductDevice(w, x, dim);
}