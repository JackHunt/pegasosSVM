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

#include "cpu_svm.h"

using namespace pegasos;

//------------------------------------------------------------------------------
//Public members.
//------------------------------------------------------------------------------

template<typename T>
cpuSVM<T>::cpuSVM(int D, T lambda) : dataDimension(D), lambda(lambda) {
    this->weights = new T[D];
    for (int i = 0; i < D; i++) {
        this->weights[i] = (T) 0.0;
    }
    this->eta = (T) 0.0;
}

template<typename T>
cpuSVM<T>::~cpuSVM() {
    delete[] weights;
}

template<typename T>
void cpuSVM<T>::train(T *data, int *labels, int instances, int batchSize) {
    std::vector<int> batch = getBatch(batchSize, instances);
    DVector<T> batchSum;

    #pragma omp parallel for
    for (std::vector<int>::iterator i = batch.begin(); i != batch.end(); ++i) {
        T inner = innerProduct(weights, data[*i]);
        if (labels[*i] * inner < 1.0) {
            DVector<T> dataVec(dataDimension, data[*i * dataDimension]);
            dataVec *= (T) labels[*i];
            #pragma omp critical 
            {
                batchSum += dataVec;
            }
        }
    }
    weightUpdate(weights, eta, lambda, batchSize, batchSum, dataDimension);
}

template<typename T>
T cpuSVM<T>::predict(T *data) {
    return innerProduct(weights, data);
}

template<typename T>
void cpuSVM<T>::predict(T* data, T* result, int instances) {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (int i = 0; i < instances; i++) {
        result[i] = innerProduct(weights, data[i * dataDimension]);
    }
}

//------------------------------------------------------------------------------
//Protected members.
//------------------------------------------------------------------------------

template<typename T>
std::vector<int> cpuSVM<T>::getBatch(int batchSize, int numElements) {
    if (batchSize < numElements) {
        std::vector<int> batchIndices(batchSize);
#if defined(_OPENMP)
#pragma omp parallel for
#endif
        for (int i = 0; i < batchSize; i++) {
            batchIndices[i] = ((rand() % batchSize));
        }
        return batchIndices;
    } else {
        std::vector<int> batchIndices(numElements);
#if defined(_OPENMP)
#pragma omp parallel for
#endif
        for (int i = 0; i < numElements; i++) {
            batchIndices[i] = i;
        }
        return batchIndices;
    }
}
