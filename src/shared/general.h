#ifndef GENERAL_SHARED_FUNCTIONS_HEADER
#define	GENERAL_SHARED_FUNCTIONS_HEADER

template<typename T>
__host__ __device__ T computeEta(T *lambda, int t){
    return (T)(1.0/(lambda*(T)t));
}

template<typename T>
__host__ __device__ T innerProduct(T *w, T*x, int dim){
    T tmp = (T)0.0;
    for(int i=0; i<dim; i++){
        tmp += w[i] * x[i];
    }
    return tmp;
}

#endif

