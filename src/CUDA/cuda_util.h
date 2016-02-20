#ifndef SVM_CUDA_UTIL_HEADER
#define	SVM_CUDA_UTIL_HEADER

#include <cuda_runtime.h>

#define CUDA_BLOCK_DIM 16
#define CUDA_CHECK(ans){cudaAssert((ans), __FILE__, __LINE__);}

inline void cudaAssert(cudaError_t code, const char *file, int line){
   if(code != cudaSuccess){
      fprintf(stderr,"CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
      exit(code);
   }
}

#endif

