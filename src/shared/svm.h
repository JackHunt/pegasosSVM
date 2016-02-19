#ifndef SVM_HEADER
#define	SVM_HEADER

#include <cmath>

namespace pegasos{
    template<typename T>
    class SVM{
    protected:
        int dataDimension;
        T eta;
        T *weights;
    
    public:
        SVM(int D, T lambda);
        virtual ~SVM() = 0;
        virtual void train(T *data, int *labels, int instances, int batchSize) = 0;
        virtual T predict(T data) = 0;
        virtual void predict(T *data, T *result, int instances) = 0;
    };
}
#endif

