#ifndef TRAINING_SHARED_HEADER
#define	TRAINING_SHARED_HEADER

template<typename T>
T computeEta(T *lambda, int t);

template<typename T>
void weightUpdate(T *weights, T eta, T lambda, int batchSize, T *batchSum, int D);

#endif

