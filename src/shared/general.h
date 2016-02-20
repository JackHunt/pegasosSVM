#ifndef GENERAL_SHARED_FUNCTIONS_HEADER
#define	GENERAL_SHARED_FUNCTIONS_HEADER

#include "shared.h"

template<typename T>
class DVector{
    private:
        T *data;
        int dim;    
    
    public:
    DVector(int dim, T *in_data = NULL) : dim(dim){
        data = new T[dim];
        for(int i=0; i<dim; i++){
            data[i] = (in_data == NULL) ? (T)0.0 : in_data[i];
        }
    }
         
    ~DVector(){
        delete[] data;
    }
    
    __SHARED_CODE__ 
    T &operator [] (int idx){
        return (idx >= dim) ? data[0] : data[idx];
    }
    
    __SHARED_CODE__
    friend DVector<T> &operator += (DVector<T> &lhs, const DVector<T> &rhs){
        for(int i=0; i<lhs->dim; i++){
            lhs[i] += rhs[i];
        }
        return lhs;
    }
    
    __SHARED_CODE__
    friend DVector<T> &operator *= (DVector<T> &lhs, const T &rhs){
        for(int i=0; i<lhs->dim; i++){
            lhs[i] *= rhs;
        }
    }
};

template<typename T>
__SHARED_CODE__
inline T innerProduct(T *A, T *B, int dim){
    T tmp = (T)0.0;
    for(int i=0; i<dim; i++){
        tmp += A[i] * B[i];
    }
    return tmp;
}

template<typename T>
__SHARED_CODE__
inline void multiply(T *A, T *B, T *out, int dim){
    T tmp = (T)0.0;
    for(int i=0; i<dim; i++){
        tmp = A[i]*B[i];
        out[i] = tmp;
    }
}

#endif