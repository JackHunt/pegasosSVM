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

#ifndef TRAINING_SHARED_HEADER
#define	TRAINING_SHARED_HEADER

#include "shared.h"

template<typename T>
__SHARED_CODE__
inline T computeEta(T *lambda, int t) {
    return (T) (1.0 / (lambda * (T) t));
}

template<typename T>
__SHARED_CODE__
inline void weightUpdateIndividual(T *weight, T *batchSum, T c1, T c2, int idx) {
    weight[idx] = c1 * weight[idx] + c2 * batchSum[idx];
}

template<typename T>
__SHARED_CODE__
inline void weightUpdate(T *weights, T eta, T lambda, int batchSize, T *batchSum, int D) {
    T c1 = (T) 1.0 - (eta * lambda);
    T c2 = eta / (T) batchSize;
    for (int i = 0; i < D; i++) {
        weightUpdateIndividual(weights, batchSum, c1, c2, i);
    }
}
#endif