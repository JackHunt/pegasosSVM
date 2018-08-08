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

#include "../shared/general.hpp"
#include "../shared/training.hpp"

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
void dotToIndicator_kernel(T *dot, int *labels, int length, T intercept) {
    int idx = getIdx();
    if (idx < 0 || idx > length - 1) return;
    dot[idx] = dotToIndicator(dot[idx], labels[idx], intercept);
}

/*
 * Intercept reduction kernel. Sums Y-X*w.
 * TO-DO: Replace this with full warp unrolling.
 */
template<typename T>
__global__
void interceptReduction_kernel(T *dot, int *labels, T *b, int batchSize) {
    __shared__ T partialSum[CUDA_BLOCK_DIM];
    int thrIdx = threadIdx.x;
    int globalIdx = getIdx();
    if (globalIdx < 0 || globalIdx > batchSize - 1) return;

    partialSum[thrIdx] = ((T)labels[globalIdx] - dot[globalIdx]);
    __syncthreads();

    for (int i = 1; i < blockDim.x; i *= 2) {
        if (thrIdx % (2 * i) == 0) {
            partialSum[thrIdx] += partialSum[thrIdx + i];
        }
        __syncthreads();
    }

    if (thrIdx == 0) {
        b[blockIdx.x] = partialSum[0];
    }
}

/*
 * Adds the intercept term to SVM outputs.
 */
template<typename T>
__global__
void addIntercept_kernel(T *outputs, T b, int length) {
    int idx = getIdx();
    if (idx < 0 || idx > length - 1) return;

    outputs[idx] += b;
}

/*
 * Kernel performing weight update for a single weight vector element.
 */
template<typename T>
__global__
void weightUpdate_kernel(T *weights, T *batchSum, T c1, T c2, int length) {
    int idx = getIdx();
    if (idx < 0 || idx > length - 1) return;
    weightUpdateIndividual(weights, batchSum, c1, c2, idx);
}
#endif