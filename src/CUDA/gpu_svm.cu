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

#include "gpu_svm.h"

using namespace pegasos;

//------------------------------------------------------------------------------
//Public members.
//------------------------------------------------------------------------------

/*
 * Initialise a gpuSVM with a given data dimension and initial lambda param.
 * Initialises a cuBLAS context, also.
 */
template<typename T>
gpuSVM<T>::gpuSVM(int D, T lambda) : dataDimension(D), lambda(lambda) {
    CUDA_CHECK(cudaMalloc((void**) & this->weights, D * sizeof (T)));
    this->eta = (T) 0.0;
    CUBLAS_CHECK(cublasCreate(&this->cublasHandle));
}

/*
 * Clean weight vector and destroy cuBLAS context.
 */
template<typename T>
gpuSVM<T>::~gpuSVM() {
    CUDA_CHECK(cudaFree(weights));
    cublasDestroy(cublasHandle);
}

/*
 * Perform a single iteration of GPU based Primal SVM training, 
 * as per pegasos paper.
 * Column wise reductions achieved by matrix multiplications, for efficiency.
 * Currently, can only process a batch in contiguous memory.
 */
template<typename T>
void gpuSVM<T>::train(T *data, int *labels, int instances, int batchSize) {
    T *dot, *reduced;
    thrust::device_ptr<T> dotP = thrust::device_pointer_cast(dot);
    thrust::device_ptr<T> reducedP = thrust::device_pointer_cast(reduced);
    CUDA_CHECK(cudaMalloc((void**) &dot, instances * sizeof (T)));
    CUDA_CHECK(cudaMalloc((void**) &reduced, dataDimension * sizeof (T)));
    if (batchSize == instances) {
        cublasMatMult(CUBLAS_OP_N, CUBLAS_OP_N, instances, 1, dataDimension,
                (T) 1.0, data, weights, (T) 0.0, dot);
        thrust::for_each(dotP, dotP + instances, dotFunctor(dot, labels, instances));
        cublasMatMult(CUBLAS_OP_T, CUBLAS_OP_T, dataDimension, 1, instances,
                (T) 1.0, data, dot, (T) 0.0, reduced);
    } else {
        throw std::invalid_argument("batchSize != instances not yet implemented!");
    }
    T c1 = (T) 1.0 - (eta * lambda);
    T c2 = eta / (T) batchSize;
    thrust::for_each(reducedP, reducedP + dataDimension, updateFunctor(weights, reduced, c1, c2));
    CUDA_CHECK(cudaFree(dot));
    CUDA_CHECK(cudaFree(reduced))
}

/*
 * Return SVM output for a given data point.
 */
template<typename T>
T gpuSVM<T>::predict(T *data) {
    return innerProduct(weights, data);
}

/*
 * Assigns SVM outputs for a given batch of instances.
 */
template<typename T>
void gpuSVM<T>::predict(T* data, T* result, int instances) {
    cublasMatMult(CUBLAS_OP_N, CUBLAS_OP_N, instances, 1, dataDimension,
            (T) 1.0, data, weights, (T) 0.0, result);
}

//------------------------------------------------------------------------------
//Protected and Private members.
//------------------------------------------------------------------------------

/*
 * CURRENTLY UNIMPLEMENTED.
 * Will generate random mini batches. Currently prohibited by performance 
 * limitations pertaining to uncoalesced read operations.
 */
template<typename T>
thrust::device_vector<int> gpuSVM<T>::getBatch(int batchSize, int numElements) {
    throw std::invalid_argument("Mini batches not yet implemented.");
}

/*
 * cuBLAS Matrix multiplication in the case of T=float
 */
template<typename T>
void gpuSVM<T>::cublasMatMult(cublasOperation_t transA, cublasOperation_t transB, int M,
        int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    int lda = (transA == CUBLAS_OP_N) ? K : M;
    int ldb = (transB == CUBLAS_OP_N) ? N : K;
    int ldc = N;
    CUBLAS_CHECK(cublasSgemm(cublasHandle, transB, transA, N, M, K, &alpha,
                B, ldb, A, lda, &beta, C, ldc));
}

/*
 * cuBLAS Matrix multiplication in the case of T=double
 */
template<typename T>
void gpuSVM<T>::cublasMatMult(cublasOperation_t transA, cublasOperation_t transB, int M,
        int N, int K, double alpha, double *A, double *B, double beta, double *C) {
    int lda = (transA == CUBLAS_OP_N) ? K : M;
    int ldb = (transB == CUBLAS_OP_N) ? N : K;
    int ldc = N;
    CUBLAS_CHECK(cublasDgemm(cublasHandle, transB, transA, N, M, K, &alpha,
                B, ldb, A, lda, &beta, C, ldc));
}