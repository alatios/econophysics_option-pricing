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

// Single stream toolbox
__host__ double GetAverage_SingleStream(double *inputStream, unsigned int streamSize){
	double streamSum = 0.;
	unsigned int streamCounter = 0;
	
	for(unsigned int RNGIndex=0; RNGIndex<streamSize; ++RNGIndex){
		streamSum += inputStream[RNGIndex];
		++streamCounter;
	}
	
	return streamSum / streamCounter;
}

__host__ double GetVariance_SingleStream(double *inputStream, double streamAverage, unsigned int streamSize){
	double streamSquaredDiscrepancyFromAverage = 0.;
	unsigned int streamCounter = 0;

	for(unsigned int RNGIndex=0; RNGIndex<streamSize; ++RNGIndex){
		streamSquaredDiscrepancyFromAverage += pow(inputStream[RNGIndex] - streamAverage,2);
		++streamCounter;
	}
	
	return streamSquaredDiscrepancyFromAverage / streamCounter;	
}

__host__ double GetKurtosis_SingleStream(double *inputStream, double streamAverage, double streamVariance, unsigned int streamSize){
	double streamQuarticDiscrepancyFromAverage = 0.;
	unsigned int streamCounter = 0;
	
	for(unsigned int RNGIndex=0; RNGIndex<streamSize; ++RNGIndex){
		streamQuarticDiscrepancyFromAverage += pow(inputStream[RNGIndex] - streamAverage,4);
		++streamCounter;
	}
	
	return streamQuarticDiscrepancyFromAverage / (streamCounter * pow(streamVariance,4));

}

__host__ double GetAutocorrelationK_SingleStream(double *inputStream, unsigned int autocorrelationOffset, unsigned int streamSize){
	double streamCorrelationSum = 0.;
	unsigned int streamCounter = 0;

	for(unsigned int RNGIndex=0; RNGIndex<streamSize-autocorrelationOffset; ++RNGIndex){
		streamCorrelationSum += inputStream[RNGIndex] * inputStream[RNGIndex+autocorrelationOffset];
		++streamCounter;
	}
	
	return streamCorrelationSum / streamCounter;
}

__host__ void EvaluateCompleteAutocorrelation_SingleStream(double *inputStream, double *outputCorrelations, unsigned int streamSize, bool verbose){
	for(unsigned int autocorrelationOffset=1; autocorrelationOffset<streamSize-1; ++autocorrelationOffset){
		outputCorrelations[autocorrelationOffset-1] = GetAutocorrelationK_SingleStream(inputStream, autocorrelationOffset, streamSize);
		
		if(verbose)
			if(autocorrelationOffset % 20 == 0)
				cout << "<xi*xi+" << autocorrelationOffset << ">:\t" << outputCorrelations[autocorrelationOffset-1] << endl;
	}
}


// Toolbox to test correlation of a stream of totalNumberOfThreads, each generating numbersToGeneratePerThread random numbers. A positive verbose variable will, well, print the output for further testing

__host__ void EvaluateAverage_MultipleStreams(double **inputStreams, double *outputStreamAverages, unsigned int totalNumberOfThreads, unsigned int numbersToGeneratePerThread, bool verbose){
	for(unsigned int threadIndex=0; threadIndex<totalNumberOfThreads; ++threadIndex){
		outputStreamAverages[threadIndex] = GetAverage_SingleStream(inputStreams[threadIndex],numbersToGeneratePerThread);	
		if(verbose)
			cout << "<xi>@thread" << threadIndex << ":\t" << outputStreamAverages[threadIndex] << endl;
	}
}

__host__ void EvaluateVariance_MultipleStreams(double **inputStreams, double *inputStreamAverages, double *outputStreamVariances, unsigned int totalNumberOfThreads, unsigned int numbersToGeneratePerThread, bool verbose){
	for(unsigned int threadIndex=0; threadIndex<totalNumberOfThreads; ++threadIndex){
		outputStreamVariances[threadIndex] = GetVariance_SingleStream(inputStreams[threadIndex], inputStreamAverages[threadIndex], numbersToGeneratePerThread);
		if(verbose)
			cout << "var(xi)@thread" << threadIndex << ":\t" << outputStreamVariances[threadIndex] << endl;
	}
}

__host__ void EvaluateKurtosis_MultipleStreams(double **inputStreams, double *inputStreamAverages, double *inputStreamVariances, double *outputStreamKurtosises, unsigned int totalNumberOfThreads, unsigned int numbersToGeneratePerThread, bool verbose){
	for(unsigned int threadIndex=0; threadIndex<totalNumberOfThreads; ++threadIndex){
		outputStreamKurtosises[threadIndex] = GetKurtosis_SingleStream(inputStreams[threadIndex], inputStreamAverages[threadIndex], inputStreamVariances[threadIndex], numbersToGeneratePerThread);
		if(verbose)
			cout << "kurt(xi)@thread" << threadIndex << ":\t" << outputStreamKurtosises[threadIndex] << endl;
	}
}

__host__ void EvaluateCompleteAutocorrelation_MultipleStreams(double **inputStreams, double **outputStreamCorrelations, unsigned int totalNumberOfThreads, unsigned int numbersToGeneratePerThread, bool verbose){
	for(unsigned int threadIndex=0; threadIndex<totalNumberOfThreads; ++threadIndex){
				
		if(verbose)
			cout << "@thread" << threadIndex << ":" << endl;

		EvaluateCompleteAutocorrelation_SingleStream(inputStreams[threadIndex], outputStreamCorrelations[threadIndex], numbersToGeneratePerThread, verbose);
	}
}
