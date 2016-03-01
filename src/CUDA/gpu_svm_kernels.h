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

#ifndef PEGASOS_GPU_SVM_KERNELS_HEADER
#define PEGASOS_GPU_SVM_KERNELS_HEADER

#include "../shared/general.h"
#include "../shared/training.h"

/*
 * Utility function, yields current threads index w.r.t the memory range that 
 * the currently executing grid is operating on. 1D block, 1D grid.
 */
__device__
inline int getIdx() {
    return blockIdx.x * blockDim.x + threadIdx.x;
}

/*
 * Dot/Inner product kernel.
 */
template<typename T>
__global__
void dotToIndicator_kernel(T *dot, int *labels, int length) {
    int idx = getIdx();
    if (idx < 0 || idx > length) return;
    dot[idx] = (dot[idx] < (T) 1.0) ? (T) labels[idx] : (T) 0.0;
}

/*
 * Kernel performing weight update for a single weight vector element.
 */
template<typename T>
__global__
void weightUpdate_kernel(T *weights, T *batchSum, T c1, T c2, int length){
    int idx = getIdx();
    if (idx < 0 || idx > length) return;
    weightUpdateIndividual(weights, batchSum, c1, c2, idx);
}
#endif