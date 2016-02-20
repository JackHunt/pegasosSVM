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

//------------------------------------------------------------------------------
//Public members.
//------------------------------------------------------------------------------
template<typename T>
gpuSVM<T>::gpuSVM(int D, T lambda) : dataDimension(D), lambda(lambda){
    CUDA_CHECK(cudaMalloc((void**)&this->weights, D*sizeof(T)));
    this->eta = (T)0.0;
}

template<typename T>
gpuSVM::~gpuSVM(){
    CUDA_CHECK(this->weights);
}

template<typename T>
void gpuSVM<T>::train(T *data, int *labels, int instances, int batchSize){
    //
}

template<typename T>
T gpuSVM<T>::predict(T *data){
    return innerProduct(weights, data);
}

template<typename T>
void gpuSVM<T>::predict(T* data, T* result, int instances){
    //
}

//------------------------------------------------------------------------------
//Protected members.
//------------------------------------------------------------------------------
template<typename T>
thrust::vector<int> gpuSVM<T>::getBatch(int batchSize, int numElements){
    if(batchSize < numElements){
        //
    }else{
        //
    }
}