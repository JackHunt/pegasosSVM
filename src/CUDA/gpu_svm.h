#ifndef PEGASOS_GPU_SVM_HEADER
#define	PEGASOS_GPU_SVM_HEADER

#include "../shared/svm.h"
#include "cuda_util.h"
#include "cuda.h"
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

namespace pegasos{
    template<typename T>
    class gpuSVM : public SVM<T>{
        protected:
            thrust::vector<int> getBatch(int batchSize, int numElements);
            int dataDimension;
            T eta, lambda;
            T *weights;
        
        public:
            gpuSVM(int D, T lambda);
            ~gpuSVM();
            void train(T *data, int *labels, int instances, int batchSize);
            T predict(T *data);
            void predict(T *data, T *result, int instances);
    };
}
#endif