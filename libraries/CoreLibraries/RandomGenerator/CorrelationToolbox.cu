#include <iostream>
#include "CorrelationToolbox.cuh"

using namespace std;

// Random number generator

__host__ void RandomNumberGeneration(unsigned int numberOfBlocks, unsigned int numberOfThreadsPerBlock, double **uniformNumbers, double **gaussianNumbers, unsigned int totalNumbersToGenerate, unsigned int numbersToGeneratePerThread, unsigned int seed){

	for(unsigned int threadNumber=0; threadNumber<numberOfBlocks*numberOfThreadsPerBlock; ++threadNumber){
		RNG *supportGenerator = new RNG_Tausworthe(seed+threadNumber);
		RNG *mainGenerator = new RNG_CombinedGenerator;
		mainGenerator->SetInternalState(supportGenerator);

		for(unsigned int RNGNumber=0; RNGNumber<numbersToGeneratePerThread; ++RNGNumber){		
			if(numbersToGeneratePerThread*threadNumber+RNGNumber < totalNumbersToGenerate){
				uniformNumbers[threadNumber][RNGNumber] = mainGenerator->GetUniform();
				gaussianNumbers[threadNumber][RNGNumber] = mainGenerator->GetGauss();
			}
		}
	}
}

// Toolbox to test correlation of a stream of totalNumberOfThreads, each generating numbersToGeneratePerThread random numbers. A positive verbose variable will, well, print the output for further testing

__host__ void EvaluateStreamAverage(double **inputStreams, double *outputStreamAverages, unsigned int totalNumberOfThreads, unsigned int numbersToGeneratePerThread, bool verbose){
	double *streamSums = new double[totalNumberOfThreads];
	unsigned int *streamCounters = new unsigned int[totalNumberOfThreads];

	for(unsigned int threadIndex=0; threadIndex<totalNumberOfThreads; ++threadIndex){
		streamSums[threadIndex] = 0.;
		streamCounters[threadIndex] = 0;
		
		for(unsigned int RNGNumber=0; RNGNumber<numbersToGeneratePerThread; ++RNGNumber){
			streamSums[threadIndex] += inputStreams[threadIndex][RNGNumber];
			++streamCounters[threadIndex];
		}
		
		outputStreamAverages[threadIndex] = streamSums[threadIndex] / streamCounters[threadIndex];
		
		if(verbose)
			cout << "<xi>@thread" << threadIndex << ":\t" << outputStreamAverages[threadIndex] << endl;
	}
	
	delete[] streamSums;
	delete[] streamCounters;
}

__host__ void EvaluateStreamVariance(double **inputStreams, double *inputStreamAverages, double *outputStreamVariances, unsigned int totalNumberOfThreads, unsigned int numbersToGeneratePerThread, bool verbose){
	double *streamSquaredDiscrepanciesFromAverage = new double[totalNumberOfThreads];
	unsigned int *streamCounters = new unsigned int[totalNumberOfThreads];
	
	for(unsigned int threadIndex=0; threadIndex<totalNumberOfThreads; ++threadIndex){
		streamSquaredDiscrepanciesFromAverage[threadIndex] = 0.;
	
		for(unsigned int RNGNumber=0; RNGNumber<numbersToGeneratePerThread; ++RNGNumber){
			streamSquaredDiscrepanciesFromAverage[threadIndex] += pow(inputStreams[threadIndex][RNGNumber] - inputStreamAverages[threadIndex],2);
			++streamCounters[threadIndex];
		}
		
		outputStreamVariances[threadIndex] = streamSquaredDiscrepanciesFromAverage[threadIndex] / streamCounters[threadIndex];

		if(verbose)
			cout << "var(xi)@thread" << threadIndex << ":\t" << outputStreamVariances[threadIndex] << endl;
	}
	
	delete[] streamSquaredDiscrepanciesFromAverage;
	delete[] streamCounters;
}

__host__ void EvaluateStreamKurtosis(double **inputStreams, double *inputStreamAverages, double *inputStreamVariances, double *outputStreamKurtosises, unsigned int totalNumberOfThreads, unsigned int numbersToGeneratePerThread, bool verbose){
	double *streamQuarticDiscrepanciesFromAverage = new double[totalNumberOfThreads];
	unsigned int *streamCounters = new unsigned int[totalNumberOfThreads];
	
	for(unsigned int threadIndex=0; threadIndex<totalNumberOfThreads; ++threadIndex){
		streamQuarticDiscrepanciesFromAverage[threadIndex] = 0.;
	
		for(unsigned int RNGNumber=0; RNGNumber<numbersToGeneratePerThread; ++RNGNumber){
			streamQuarticDiscrepanciesFromAverage[threadIndex] += pow(inputStreams[threadIndex][RNGNumber] - inputStreamAverages[threadIndex],4);
			++streamCounters[threadIndex];
		}
		
		outputStreamKurtosises[threadIndex] = streamQuarticDiscrepanciesFromAverage[threadIndex] / (streamCounters[threadIndex] * pow(inputStreamVariances[threadIndex],4));
		if(verbose)
			cout << "kurt(xi)@thread" << threadIndex << ":\t" << outputStreamKurtosises[threadIndex] << endl;
	}
	
	delete[] streamQuarticDiscrepanciesFromAverage;
	delete[] streamCounters;
}

__host__ void EvaluateSingleAutocorrelation(double **inputStreams, double *outputStreamCorrelations, unsigned int autocorrelationOffset, unsigned int totalNumberOfThreads, unsigned int numbersToGeneratePerThread, bool verbose){
	double *streamCorrelationSums = new double[totalNumberOfThreads];
	unsigned int *streamCorrelationCounters = new unsigned int[totalNumberOfThreads];
	
	for(unsigned int threadIndex=0; threadIndex<totalNumberOfThreads; ++threadIndex){
		streamCorrelationSums[threadIndex] = 0.;
		streamCorrelationCounters[threadIndex] = 0;
		
		for(unsigned int RNGNumber=0; RNGNumber<numbersToGeneratePerThread-autocorrelationOffset; ++RNGNumber){
			streamCorrelationSums[threadIndex] += inputStreams[threadIndex][RNGNumber] * inputStreams[threadIndex][RNGNumber+autocorrelationOffset];
			++streamCorrelationCounters[threadIndex];
		}
		
		outputStreamCorrelations[threadIndex] = streamCorrelationSums[threadIndex] / streamCorrelationCounters[threadIndex];
		
		if(verbose)
			if(threadIndex % 100 == 0)
				cout << "<xi*xi+" << autocorrelationOffset << ">@thread" << threadIndex << ":\t" << outputStreamCorrelations[threadIndex] << endl;
	}
	
	delete[] streamCorrelationSums;
	delete[] streamCorrelationCounters;
}

__host__ void EvaluateCompleteAutocorrelation(double **inputStreams, double *outputStreamCorrelations, unsigned int totalNumberOfThreads, unsigned int numbersToGeneratePerThread, bool verbose){
	for(unsigned int autocorrelationOffset=1; autocorrelationOffset<numbersToGeneratePerThread-1; ++autocorrelationOffset){
		if(verbose)
			if(autocorrelationOffset % 20 == 0){
				EvaluateSingleAutocorrelation(inputStreams, outputStreamCorrelations, autocorrelationOffset, totalNumberOfThreads, numbersToGeneratePerThread, true);
				continue;
			}
		
		EvaluateSingleAutocorrelation(inputStreams, outputStreamCorrelations, autocorrelationOffset, totalNumberOfThreads, numbersToGeneratePerThread, false);
	}
}
