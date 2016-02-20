#ifndef SHARED_CODE_HEADER
#define SHARED_CODE_HEADER

#if defined(__CUDACC__) && defined(__CUDA_ARCH__)
    #define __SHARED_CODE__ __device__
#else
    #define __SHARED_CODE__ 
#endif

#endif

