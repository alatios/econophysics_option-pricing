#ifndef _CORRELATION__TOOLBOX_CUH
#define _CORRELATION__TOOLBOX_CUH

#include "RNG.cuh"

// Random number generator
__host__ void RandomNumberGeneration(unsigned int numberOfBlocks, unsigned int numberOfThreadsPerBlock, double **uniformNumbers, double **gaussianNumbers, unsigned int totalNumbersToGenerate, unsigned int numbersToGeneratePerThread, unsigned int seed);

// Toolbox to test correlation of a stream of totalNumberOfThreads, each generating numbersToGeneratePerThread random numbers. A positive verbose variable will, well, print the output for further testing
__host__ void EvaluateStreamAverage(double **inputStreams, double *outputStreamAverages, unsigned int totalNumberOfThreads, unsigned int numbersToGeneratePerThread, bool verbose);
__host__ void EvaluateStreamVariance(double **inputStreams, double *inputStreamAverages, double *outputStreamVariances, unsigned int totalNumberOfThreads, unsigned int numbersToGeneratePerThread, bool verbose);
__host__ void EvaluateStreamKurtosis(double **inputStreams, double *inputStreamAverages, double *inputStreamVariances, double *outputStreamKurtosises, unsigned int totalNumberOfThreads, unsigned int numbersToGeneratePerThread, bool verbose);

__host__ void EvaluateSingleAutocorrelation(double **inputStreams, double *outputStreamCorrelations, double autocorrelationOffset, unsigned int totalNumberOfThreads, unsigned int numbersToGeneratePerThread, bool verbose);
__host__ void EvaluateCompleteAutocorrelation(double **inputStreams, double *outputStreamCorrelations, unsigned int totalNumberOfThreads, unsigned int numbersToGeneratePerThread, bool verbose);
#endif
