#ifndef CPU_SVM_HEADER
#define	CPU_SVM_HEADER

#include "../shared/svm.h"
//#include <blas.h>

namespace pegasos{
    template<typename T>
    class cpuSVM : public SVM<T>{
        protected:
            int dataDimension;
            float eta;
            T *weights;
        
        public:
            cpuSVM(int D, T lambda);
            ~cpuSVM();
            void train(T *data, int *labels, int instances, int batchSize);
            T predict(T data);
            void predict(T *data, T *result, int instances);
    };
}

#endif

