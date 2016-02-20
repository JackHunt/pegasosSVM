#ifndef PEGASOS_CPU_SVM_HEADER
#define	PEGASOS_CPU_SVM_HEADER

#include "../shared/svm.h"
#include <cstdlib>
#include <vector>

namespace pegasos{
    template<typename T>
    class cpuSVM : public SVM<T>{
        protected:
            std::vector<int> getBatch(int batchSize, int numElements);
            int dataDimension;
            T eta, lambda;
            T *weights;
        
        public:
            cpuSVM(int D, T lambda);
            ~cpuSVM();
            void train(T *data, int *labels, int instances, int batchSize);
            T predict(T *data);
            void predict(T *data, T *result, int instances);
    };
}
#endif

