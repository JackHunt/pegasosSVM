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

#include "cpu_svm.hpp"

using namespace pegasos;

//------------------------------------------------------------------------------
//Public members.
//------------------------------------------------------------------------------

/*
 * Initialise the SVM.
 */
template<typename T>
cpuSVM<T>::cpuSVM(int D, T lambda) : dataDimension(D), lambda(lambda), b(0.0) {
    this->weights = new T[D];
    this->reset();
}

/*
 * Deallocates temporary buffers.
 */
template<typename T>
cpuSVM<T>::~cpuSVM() {
    delete[] weights;
}

/*
 * Performs batch training.
 */
template<typename T>
void cpuSVM<T>::train(T *data, int *labels, int instances, int batchSize) {
    if (batchSize != instances) {
        throw std::invalid_argument("batchSize != instances not yet implemented!");
    }
    T *dot, *reduced;
    dot = new T[batchSize];
    reduced = new T[dataDimension];
    blasMatVecMult(CblasNoTrans, batchSize, dataDimension, (T) 1.0, data, weights, (T) 0.0, dot);

    b = 0.0;
#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (int i = 0; i < batchSize; i++) {
        b += (labels[i] - dot[i]);
    }
    b /= (T)batchSize;

#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (int i = 0; i < batchSize; i++) {
        dot[i] = dotToIndicator(dot[i], labels[i], b);
    }
    blasMatVecMult(CblasTrans, batchSize, dataDimension, (T) 1.0, data, dot, (T) 0.0, reduced);

    weightUpdate(weights, eta, lambda, batchSize, reduced, dataDimension);
    T norm = l2Norm(weights, dataDimension);
    if (sqrt(lambda) * norm < 1) {
        //weightProjection(weights, lambda, norm, dataDimension);
    }
    eta = computeEta<T>(lambda, timeStep);
    timeStep++;

    delete[] dot;
    delete[] reduced;
}

/*
 * Return SVM output for a given data point.
 */
template<typename T>
T cpuSVM<T>::predict(T *data) {
    return innerProduct(weights, data, dataDimension) + b;
}

/*
 * Assigns SVM outputs for a given batch of instances.
 */
template<typename T>
void cpuSVM<T>::predict(T* data, T* result, int instances) {
    blasMatVecMult(CblasNoTrans, instances, dataDimension, (T) 1.0, data, weights, (T) 0.0, result);
    for (int i = 0; i < instances; i++) {
        result[i] += b;
    }
}

/*
 * Reset the SVM.
 */
template<typename T>
void cpuSVM<T>::reset() {
    this->timeStep = 1;
    this->eta = (T) 0.0;
    srand(time(0));
    for (int i = 0; i < this->dataDimension; i++) {
        this->weights[i] = (T)rand() / (T)RAND_MAX;
    }
    T norm = l2Norm(this->weights, this->dataDimension);
    //weightProjection(this->weights, this->lambda, norm, this->dataDimension);
}

//------------------------------------------------------------------------------
//Protected and Private members.
//------------------------------------------------------------------------------

/*
 * CURRENTLY UNIMPLEMENTED.
 */
template<typename T>
std::vector<int> cpuSVM<T>::getBatch(int batchSize, int numElements) {
    throw std::invalid_argument("Mini batches not yet implemented.");
}

/*
 * BLAS Matrix-Vector multiplication in the case of T=float
 */
template<typename T>
void cpuSVM<T>::blasMatVecMult(CBLAS_TRANSPOSE transA, int M, int N, float alpha, float *A,
                               float *x, float beta, float *C) {
    cblas_sgemv(CblasRowMajor, transA, M, N, alpha, A, N, x, 1, beta, C, 1);
}

/*
 * BLAS Matrix-Vector multiplication in the case of T=double
 */
template<typename T>
void cpuSVM<T>::blasMatVecMult(CBLAS_TRANSPOSE transA, int M, int N, double alpha, double *A,
                               double *x, double beta, double *C) {
    cblas_dgemv(CblasRowMajor, transA, M, N, alpha, A, N, x, 1, beta, C, 1);
}

template class cpuSVM<float>;
template class cpuSVM<double>;