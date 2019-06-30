#ifndef _CORRELATION__TOOLBOX_CUH
#define _CORRELATION__TOOLBOX_CUH

#include "RNG.cuh"

// Random number generator
__host__ void RandomNumberGeneration(unsigned int numberOfBlocks, unsigned int numberOfThreadsPerBlock, double **uniformNumbers, double **gaussianNumbers, unsigned int totalNumbersToGenerate, unsigned int numbersToGeneratePerThread, unsigned int seed);

// Single stream toolbox. A true verbose variable will print the output for further testing
__host__ double GetAverage_SingleStream(double *inputStream, unsigned int streamSize);
__host__ double GetVariance_SingleStream(double *inputStream, double streamAverage, unsigned int streamSize);
__host__ double GetKurtosis_SingleStream(double *inputStream, double streamAverage, double streamVariance, unsigned int streamSize);
__host__ double GetAutocorrelationK_SingleStream(double *inputStream, unsigned int autocorrelationOffset, unsigned int streamSize);
	// Accepts two vectors: one is streamSize sized, the other is streamSize-2
__host__ void EvaluateCompleteAutocorrelation_SingleStream(double *inputStream, double *outputCorrelations, unsigned int streamSize, bool verbose);


// Many-streams toolbox. A true verbose variable will print the output for further testing
__host__ void EvaluateAverage_MultipleStreams(double **inputStreams, double *outputStreamAverages, unsigned int totalNumberOfThreads, unsigned int numbersToGeneratePerThread, bool verbose);
__host__ void EvaluateVariance_MultipleStreams(double **inputStreams, double *inputStreamAverages, double *outputStreamVariances, unsigned int totalNumberOfThreads, unsigned int numbersToGeneratePerThread, bool verbose);
__host__ void EvaluateKurtosis_MultipleStreams(double **inputStreams, double *inputStreamAverages, double *inputStreamVariances, double *outputStreamKurtosises, unsigned int totalNumberOfThreads, unsigned int numbersToGeneratePerThread, bool verbose);
	// Accepts two matrixes with totalNumberOfThreads as first index and numbersToGeneratePerThread or numbersToGeneratePerThread-2 as second index respectively
__host__ void EvaluateCompleteAutocorrelation_MultipleStreams(double **inputStreams, double **outputStreamCorrelations, unsigned int totalNumberOfThreads, unsigned int numbersToGeneratePerThread, bool verbose);
#endif
