//
//	INTER- AND INTRA-STREAM CORRELATION TEST
//
//	Tests correlation of random generated numbers within single streams
//	as well as across all streams.
//

#include <iostream>
#include <ctime>		// time(NULL) for seed
#include <climits>		// UINT_MAX

#include "RNG.cuh"
#include "CorrelationToolbox.cuh"

using namespace std;

int main(){
	
	// Verbosities
	bool intraStreamAverageVerbosity = false;
	bool intraStreamVarianceVerbosity = false;
	bool intraStreamKurtosisVerbosity = false;
	bool intraStreamAutocorrelationVerbosity = false;
	
	bool interStreamAverageVerbosity = true;
	bool interStreamVarianceVerbosity = true;
	bool interStreamKurtosisVerbosity = true;
	bool interStreamAutocorrelationVerbosity = true;
	
	
	unsigned int numberOfBlocks = 1;
	unsigned int numberOfThreadsPerBlock = 512;
	unsigned int totalNumberOfThreads = numberOfBlocks * numberOfThreadsPerBlock;
	unsigned int numbersToGeneratePerThread = 1000;
	unsigned int totalNumbersToGenerate = totalNumberOfThreads * numbersToGeneratePerThread;

	cout << "Total numbers to generate: " << totalNumbersToGenerate << endl;
	cout << "Total number of threads: " << totalNumberOfThreads << endl;
	cout << "Total numbers to generate per thread: " << numbersToGeneratePerThread << endl;
	
	double **uniformNumbers = new double*[totalNumberOfThreads];
	double **gaussianNumbers = new double*[totalNumberOfThreads];
	
	for(unsigned int threadIndex=0; threadIndex<totalNumberOfThreads; ++threadIndex){
		uniformNumbers[threadIndex] = new double[numbersToGeneratePerThread];
		gaussianNumbers[threadIndex] = new double[numbersToGeneratePerThread];
	}

	// Seed for Tausworthe support generator
	unsigned int seed;	
	do
		seed = time(NULL);
	while(seed < 129 || seed > UINT_MAX - totalNumberOfThreads);
	

	RandomNumberGeneration(numberOfBlocks, numberOfThreadsPerBlock, uniformNumbers, gaussianNumbers, totalNumbersToGenerate, numbersToGeneratePerThread, seed);
	
	cout << endl <<  "############### INTRA-STREAM TEST ###############" << endl;
	
	// Average, exp. 0.5 for uniform numbers, 0 for gaussian numbers
	double *uniformAverages = new double[totalNumberOfThreads];
	double *gaussAverages = new double[totalNumberOfThreads];
	
	if(intraStreamAverageVerbosity)
		cout << endl << "Uniform averages (exp. 0.5):" << endl;
	EvaluateAverage_MultipleStreams(uniformNumbers, uniformAverages, totalNumberOfThreads, numbersToGeneratePerThread, intraStreamAverageVerbosity);
	
	if(intraStreamAverageVerbosity)
		cout << endl << "Gaussian averages (exp. 0):" << endl;
	EvaluateAverage_MultipleStreams(gaussianNumbers, gaussAverages, totalNumberOfThreads, numbersToGeneratePerThread, intraStreamAverageVerbosity);

	// Gaussian variance, exp. 1
	double *gaussVariances = new double[totalNumberOfThreads];
	if(intraStreamVarianceVerbosity)
		cout << endl << "Gaussian variances (exp. 1):" << endl;
	EvaluateVariance_MultipleStreams(gaussianNumbers, gaussAverages, gaussVariances, totalNumberOfThreads, numbersToGeneratePerThread, intraStreamVarianceVerbosity);
	
	// Gaussian kurtosis, exp. 3
	double *gaussKurtosises = new double[totalNumberOfThreads];
	if(intraStreamKurtosisVerbosity)
		cout << endl << "Gaussian kurtosises (exp. 3):" << endl;
	EvaluateKurtosis_MultipleStreams(gaussianNumbers, gaussAverages, gaussVariances, gaussKurtosises, totalNumberOfThreads, numbersToGeneratePerThread, intraStreamKurtosisVerbosity);

	// Autocorrelation i/i+k, exp. 0.25 for uniform numbers, 0 for gaussian numbers
	double **uniformCorrelations = new double*[totalNumberOfThreads];
	double **gaussCorrelations = new double*[totalNumberOfThreads];
	
	for(unsigned int threadIndex=0; threadIndex<totalNumberOfThreads; ++threadIndex){
		// One component for each autocorrelation offset k (as in xi*xi+k)
		// which runs from 1 to numbersToGeneratePerThread-1
		uniformCorrelations[threadIndex] = new double[numbersToGeneratePerThread-2];	
		gaussCorrelations[threadIndex] = new double[numbersToGeneratePerThread-2];
	}

	if(intraStreamAutocorrelationVerbosity)
		cout << endl << "Uniform autocorrelations i/i+k (exp. 0.25):" << endl;
	EvaluateCompleteAutocorrelation_MultipleStreams(uniformNumbers, uniformCorrelations, totalNumberOfThreads, numbersToGeneratePerThread, intraStreamAutocorrelationVerbosity);
	
	if(intraStreamAutocorrelationVerbosity)
		cout << endl << "Gaussian autocorrelations i/i+k (exp. 0):" << endl;
	EvaluateCompleteAutocorrelation_MultipleStreams(gaussianNumbers, gaussCorrelations, totalNumberOfThreads, numbersToGeneratePerThread, intraStreamAutocorrelationVerbosity);

///////////////////////////////////////////////////////////////////////////////////////////////////////

	cout << endl <<  "############### INTER-STREAM TEST ###############" << endl;
	// Build the Super Stream, i.e. the combination of all streams of RNGs
	double *superStreamOfUniformNumbers = new double[totalNumbersToGenerate];
	double *superStreamOfGaussianNumbers = new double[totalNumbersToGenerate];
	
	for(unsigned int RNGIndex=0; RNGIndex<numbersToGeneratePerThread; ++RNGIndex){
		for(unsigned int threadIndex=0; threadIndex<totalNumberOfThreads; ++threadIndex){
			superStreamOfUniformNumbers[RNGIndex*totalNumberOfThreads+threadIndex] = uniformNumbers[threadIndex][RNGIndex];
			superStreamOfGaussianNumbers[RNGIndex*totalNumberOfThreads+threadIndex] = gaussianNumbers[threadIndex][RNGIndex];
		}
	}
	
	// Average, exp. 0.5 for uniform numbers, 0 for gaussian numbers
	double superStreamUniformAverage = GetAverage_SingleStream(superStreamOfUniformNumbers, totalNumbersToGenerate);
	double superStreamGaussAverage = GetAverage_SingleStream(superStreamOfGaussianNumbers, totalNumbersToGenerate);
	if(interStreamAverageVerbosity){
		cout << "Uniform super stream avg (exp. 0.5):\t" << superStreamUniformAverage << endl;
		cout << "Gaussian super stream avg (exp. 0):\t" << superStreamGaussAverage << endl;
	}
	
	// Gaussian variance, exp. 1
	double superStreamGaussianVariance = GetVariance_SingleStream(superStreamOfGaussianNumbers, superStreamGaussAverage, totalNumbersToGenerate);
	if(interStreamVarianceVerbosity){
		cout << "Gaussian super stream variance (exp. 1):\t" << superStreamGaussianVariance << endl;
	}
	
	// Gaussian kurtosis, exp. 3
	double superStreamGaussianKurtosis = GetKurtosis_SingleStream(superStreamOfGaussianNumbers, superStreamGaussAverage, superStreamGaussianVariance, totalNumbersToGenerate);
	if(interStreamKurtosisVerbosity){
		cout << "Gaussian super stream kurtosis (exp. 3):\t" << superStreamGaussianKurtosis << endl;
	}
	
	// Autocorrelation i/i+k, exp. 0.25 for uniform numbers, 0 for gaussian numbers
	double *superStreamUniformCorrelations = new double[totalNumbersToGenerate-2];
	double *superStreamGaussianCorrelations = new double[totalNumbersToGenerate-2];
	if(interStreamAutocorrelationVerbosity)
		cout << "Uniform super stream autocorrelation (exp. 0.25):" << endl;
	EvaluateCompleteAutocorrelation_SingleStream(superStreamOfUniformNumbers, superStreamUniformCorrelations, totalNumbersToGenerate, interStreamAutocorrelationVerbosity);
	if(interStreamAutocorrelationVerbosity)
		cout << "Gaussian super stream autocorrelation (exp. 0):" << endl;
	EvaluateCompleteAutocorrelation_SingleStream(superStreamOfGaussianNumbers, superStreamGaussianCorrelations, totalNumbersToGenerate, interStreamAutocorrelationVerbosity);

///////////////////////////////////////////////////////////////////////////////////////////////////////

	// Quite a lot of space to free up
	delete[] uniformAverages;
	delete[] gaussAverages;
	delete[] gaussVariances;
	delete[] gaussKurtosises;

	for(unsigned int threadIndex=0; threadIndex<totalNumberOfThreads; ++threadIndex){
		delete[] uniformCorrelations[threadIndex];
		delete[] gaussCorrelations[threadIndex];
		delete[] uniformNumbers[threadIndex];
		delete[] gaussianNumbers[threadIndex];
	}
	
	delete[] uniformCorrelations;
	delete[] gaussCorrelations;
	delete[] superStreamUniformCorrelations;
	delete[] superStreamGaussianCorrelations;

	delete[] superStreamOfUniformNumbers;
	delete[] superStreamOfGaussianNumbers;
	delete[] uniformNumbers;
	delete[] gaussianNumbers;
	
	return 0;

}
