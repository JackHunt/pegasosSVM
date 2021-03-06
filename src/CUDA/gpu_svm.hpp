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

#ifndef PEGASOS_GPU_SVM_HEADER
#define PEGASOS_GPU_SVM_HEADER

#include "cuda.h"
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/execution_policy.h>
#include <stdexcept>

#include "../shared/svm.hpp"
#include "cuda_util.hpp"
#include "gpu_svm_kernels.hpp"

 /*
  * gpuSVM class, derived from SVM defined in shared/svm.h
  */
namespace pegasos {

    template<typename T>
    class gpuSVM : public SVM<T> {
    private:
        cublasHandle_t cublasHandle;
        void cublasMatVecMult(cublasOperation_t transA, int M, int N, float alpha, float *A,
                              float *x, float beta, float *C);
        void cublasMatVecMult(cublasOperation_t transA, int M, int N, double alpha, double *A,
                              double *x, double beta, double *C);
        void setIntercept(T *dot, int *labels, int batchSize);

    protected:
        thrust::device_vector<int> getBatch(int batchSize, int numElements);
        int dataDimension, timeStep;
        T eta, lambda, b;
        T *weights;

    public:
        gpuSVM(int D, T lambda);
        ~gpuSVM();
        void train(T *data, int *labels, int instances, int batchSize);
        T predict(T *data);
        void predict(T *data, T *result, int instances);
        void reset();
    };
}
#endif
