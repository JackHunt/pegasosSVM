/*
 Copyright (c) 2016, Jack Miles Hunt
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
 * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
 * Neither the name of Jack Miles Hunt nor the
      names of contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL Jack Miles Hunt BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef GENERAL_SHARED_FUNCTIONS_HEADER
#define	GENERAL_SHARED_FUNCTIONS_HEADER

#include "shared.h"

template<typename T>
class DVector {
private:
    T *data;
    int dim;

public:

    DVector(int dim, T *in_data = NULL) : dim(dim) {
        data = new T[dim];
        for (int i = 0; i < dim; i++) {
            data[i] = (in_data == NULL) ? (T) 0.0 : in_data[i];
        }
    }

    ~DVector() {
        delete[] data;
    }

    __SHARED_CODE__
    T &operator[](int idx) {
        return (idx >= dim) ? data[0] : data[idx];
    }

    __SHARED_CODE__
            friend DVector<T> &operator+=(DVector<T> &lhs, const DVector<T> &rhs) {
        for (int i = 0; i < lhs->dim; i++) {
            lhs[i] += rhs[i];
        }
        return lhs;
    }

    __SHARED_CODE__
            friend DVector<T> &operator*=(DVector<T> &lhs, const T &rhs) {
        for (int i = 0; i < lhs->dim; i++) {
            lhs[i] *= rhs;
        }
    }
};

template<typename T>
__SHARED_CODE__
inline T innerProduct(T *A, T *B, int dim) {
    T tmp = (T) 0.0;
    for (int i = 0; i < dim; i++) {
        tmp += A[i] * B[i];
    }
    return tmp;
}

template<typename T>
__SHARED_CODE__
inline void multiply(T *A, T *B, T *out, int dim) {
    T tmp = (T) 0.0;
    for (int i = 0; i < dim; i++) {
        tmp = A[i] * B[i];
        out[i] = tmp;
    }
}

#endif