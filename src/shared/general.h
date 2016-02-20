#ifndef GENERAL_SHARED_FUNCTIONS_HEADER
#define	GENERAL_SHARED_FUNCTIONS_HEADER

#include "shared.h"

template<typename T>
__SHARED_CODE__
inline T innerProduct(T *w, T *x, int dim){
    T tmp = (T)0.0;
    for(int i=0; i<dim; i++){
        tmp += w[i] * x[i];
    }
    return tmp;
}

#endif

