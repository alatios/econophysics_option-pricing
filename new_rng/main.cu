#include <iostream>
#include <cstdlib>
#include <ctime>		// time(NULL) for seed
#include <random>		// C++11 Mersenne twister
#include <climits>		// UINT_MAX
#include <cmath>		// log, cos, sin, ceil, M_PI
#include <algorithm>	// min
#include <fstream>
#include <cstdio>
#include <vector>
#include "rng.cuh"

using namespace std;

__host__ __device__ void GenerateRandomNumbers_HostDev(RandomNumberGenerator **generators, unsigned int *unsignedNumbers, double *uniformNumbers, double *gaussianNumbers, unsigned int totalNumbersToGenerate, int unsigned int numbersToGeneratePerThread, unsigned int threadNumber);
__host__ void GenerateRandomNumbers_Host(RandomNumberGenerator **generators, unsigned int *unsignedNumbers, double *uniformNumbers, double *gaussianNumbers, unsigned int totalNumbersToGenerate, int unsigned int numbersToGeneratePerThread, unsigned int numberOfBlocks, unsigned int numberOfThreadsPerBlock);


int main(){
	
	unsigned int numberOfBlocks = 2;
	unsigned int numberOfThreadsPerBlock = 512;
	unsigned int totalNumberOfThreads = numberOfBlocks * numberOfThreadsPerBlock;
	
	unsigned int totalNumbersToGenerate = 10;
	unsigned int numbersToGeneratePerThread = ceil(static_cast<double>(totalNumbersToGenerate) / totalNumberOfThreads);
	
	mt19937 mersenneCoreGenerator(time(NULL));
	uniform_int_distribution<unsigned int> mersenneDistribution(129, UINT_MAX);
	
	cout << "Numbers to generate: " << totalNumbersToGenerate << endl;
	cout << "Total number of threads: " << totalNumberOfThreads << endl;
	cout << "Numbers each thread generates (round up): " << numbersToGeneratePerThread << endl;
	
	RandomNumberGenerator** generators = new RandomNumberGenerator*[totalNumberOfThreads];
	
	for(unsigned int threadNumber=0; threadNumber<totalNumberOfThreads; ++threadNumber){
		generators[threadNumber] = new RandomNumberGenerator_Hybrid(
												mersenneDistribution(mersenneCoreGenerator),
												mersenneDistribution(mersenneCoreGenerator),
												mersenneDistribution(mersenneCoreGenerator),
												mersenneDistribution(mersenneCoreGenerator)
											);
	}

//	for(int i=0; i<totalNumbersToGenerate; ++i)
//		cout << i << " " << generator->GetUniform() << " " << generator->GetGauss() << endl;
		
	unsigned int *unsignedNumbers = new unsigned int[totalNumbersToGenerate];
	double *uniformNumbers = new double[totalNumbersToGenerate];
	double *gaussianNumbers = new double[totalNumbersToGenerate];

///*
	////////////// HOST-SIDE GENERATOR //////////////
	GenerateRandomNumbers_Host(generators, unsignedNumbers, uniformNumbers, totalNumbersToGenerate, numbersToGeneratePerThread, numberOfBlocks, numberOfThreadsPerBlock);
	/////////////////////////////////////////////////
//*/


	for(int randomNumber=0; randomNumber<totalNumbersToGenerate; ++randomNumber){
		cout << randomNumber << "\t" << unsignedNumbers << "\t" << uniformNumbers[randomNumber] << "\t" << gaussianNumbers[randomNumber] << endl;
	}

	
	for(threadNumber=0; threadNumber<totalNumberOfThreads; ++threadNumber)
		delete generators[threadNumber];
		
	delete[] generators;
	delete[] unsignedNumbers;
	delete[] uniformNumbers;
	delete[] gaussianNumbers;

	return 0;
}

__host__ __device__ void GenerateRandomNumbers_HostDev(RandomNumberGenerator **generators, unsigned int *unsignedNumbers, double *uniformNumbers, double *gaussianNumbers, unsigned int totalNumbersToGenerate, int unsigned int numbersToGeneratePerThread, unsigned int threadNumber){
	
	unsigned int temp_unsigned;
	double temp_gaussian, temp_uniform;
	for(unsigned int RNGNumber=0; RNGNumber<numbersToGeneratePerThread; ++RNGNumber){
		if(numbersToGeneratePerThread*threadNumber+RNGNumber < totalNumbersToGenerate){
			temp_unsigned = generators[threadNumber]->GetUnsignedInt();		
			unsignedNumbers[numbersToGeneratePerThread*threadNumber+RNGNumber] = temp_unsigned;
			
			temp_uniform = generators[threadNumber]->GetUniform();
			uniformNumbers[numbersToGeneratePerThread*threadNumber+RNGNumber] = temp_uniform;
			
			temp_gaussian = generators[threadNumber]->GetGauss();
			gaussianNumbers[numbersToGeneratePerThread*threadNumber+RNGNumber] = temp_gaussian;
		}
	}
}

__host__ void GenerateRandomNumbers_Host(RandomNumberGenerator **generators, unsigned int *unsignedNumbers, double *uniformNumbers, double *gaussianNumbers, unsigned int totalNumbersToGenerate, int unsigned int numbersToGeneratePerThread, unsigned int numberOfBlocks, unsigned int numberOfThreadsPerBlock){
	
	for(unsigned int threadNumber=0; threadNumber<numberOfBlocks*numberOfThreadsPerBlock; ++threadNumber){
		GenerateRandomNumbers_HostDev(generators, numbersToGeneratePerThread, uniformNumbers, gaussianNumbers, threadNumber, numberOfBlocks, numberOfThreadsPerBlock, totalNumbersToGenerate);
	}
}
