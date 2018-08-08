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

#include "gpu_svm.hpp"

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
    CUDA_CHECK(cudaMalloc((void**)&this->weights, dataDimension * sizeof(T)));
    this->reset();
    CUBLAS_CHECK(cublasCreate_v2(&this->cublasHandle));
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
    if (batchSize != instances) {
        throw std::invalid_argument("batchSize != instances not yet implemented!");
    }

    T *dot, *reduced;
    CUDA_CHECK(cudaMalloc((void**)&dot, batchSize * sizeof(T)));
    CUDA_CHECK(cudaMalloc((void**)&reduced, dataDimension * sizeof(T)));

    cublasMatVecMult(CUBLAS_OP_T, batchSize, dataDimension, (T) 1.0, data, weights, (T) 0.0, dot);
    int gridDimension = (int)ceil((float)batchSize / (float)CUDA_BLOCK_DIM);
    dotToIndicator_kernel << <gridDimension, CUDA_BLOCK_DIM >> > (dot, labels, batchSize, b);

    setIntercept(dot, labels, batchSize);

    cublasMatVecMult(CUBLAS_OP_N, batchSize, dataDimension, (T) 1.0, data, dot, (T) 0.0, reduced);

    T c1 = computeCoeff1<T>(eta, lambda);
    T c2 = computeCoeff2<T>(eta, batchSize);
    gridDimension = (int)ceil((float)dataDimension / (float)CUDA_BLOCK_DIM);
    weightUpdate_kernel << <gridDimension, CUDA_BLOCK_DIM >> > (weights, reduced, c1, c2, dataDimension);

    eta = computeEta<T>(lambda, timeStep);
    timeStep++;

    CUDA_CHECK(cudaFree(dot));
    CUDA_CHECK(cudaFree(reduced));
}

/*
 * Return SVM output for a given data point.
 */
template<typename T>
T gpuSVM<T>::predict(T *data) {
    T weights_cpu[dataDimension];
    T data_cpu[dataDimension];
    CUDA_CHECK(cudaMemcpy(weights_cpu, weights, dataDimension * sizeof(T), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(data_cpu, data, dataDimension * sizeof(T), cudaMemcpyDeviceToHost));
    return innerProduct(weights_cpu, data_cpu, dataDimension) + b;
}

/*
 * Assigns SVM outputs for a given batch of instances.
 */
template<typename T>
void gpuSVM<T>::predict(T* data, T* result, int instances) {
    cublasMatVecMult(CUBLAS_OP_T, instances, dataDimension, (T) 1.0, data, weights, (T) 0.0, result);
    int gridDimension = (int)ceil((float)instances / (float)CUDA_BLOCK_DIM);
    addIntercept_kernel << <gridDimension, CUDA_BLOCK_DIM >> > (result, b, instances);

    T tmp[instances];
    cudaMemcpy(&tmp, result, instances * sizeof(T), cudaMemcpyDeviceToHost);
    for (int i = 0; i < instances; i++) {
        std::cout << tmp[i] << std::endl;
    }
}

/*
 * Randomises the weight vector and resets eta, the time step and the intercept.
 */
template<typename T>
void gpuSVM<T>::reset() {
    this->eta = (T)0.0;
    this->timeStep = 1;
    this->b = (T)0.0;
    srand(time(0));
    T tmp[dataDimension];
    for (int i = 0; i < dataDimension; i++) {
        tmp[i] = (T)rand() / (T)RAND_MAX;
    }
    CUDA_CHECK(cudaMemcpy(this->weights, &tmp, dataDimension * sizeof(T), cudaMemcpyHostToDevice));
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
 * Update intercept term w.r.t the current data and weights.
 */
template<typename T>
void gpuSVM<T>::setIntercept(T *dot, int *labels, int batchSize) {
    int gridDimension = (int)ceil((float)batchSize / (float)CUDA_BLOCK_DIM);
    thrust::device_vector<T> partialSums(gridDimension, 0.0);
    interceptReduction_kernel << <gridDimension, CUDA_BLOCK_DIM >> > (dot, labels, thrust::raw_pointer_cast(partialSums.data()), batchSize);
    cudaDeviceSynchronize();
    b = thrust::reduce(partialSums.begin(), partialSums.end()) / batchSize;
}

/*
 * cuBLAS Matrix-Vector multiplication in the case of T=float
 */
template<typename T>
void gpuSVM<T>::cublasMatVecMult(cublasOperation_t transA, int M, int N, float alpha, float *A,
                                 float *x, float beta, float *C) {
    CUBLAS_CHECK(cublasSgemv_v2(cublasHandle, transA, M, N, &alpha, A, M, x, 1, &beta, C, 1));
}

/*
 * cuBLAS Matrix-Vector multiplication in the case of T=double
 */
template<typename T>
void gpuSVM<T>::cublasMatVecMult(cublasOperation_t transA, int M, int N, double alpha, double *A,
                                 double *x, double beta, double *C) {
    CUBLAS_CHECK(cublasDgemv_v2(cublasHandle, transA, M, N, &alpha, A, N, x, 1, &beta, C, 1));
}
template class gpuSVM<float>;
template class gpuSVM<double>;
