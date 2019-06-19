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

__global__ void RNGen_Global(RNGCombinedGenerator *generators, unsigned int *unsignedNumbers, double *uniformNumbers, double *gaussianNumbers, unsigned int totalNumbersToGenerate, unsigned int numbersToGeneratePerThread);
__host__ __device__ void RNGen_HostDev(RNGCombinedGenerator *generators, unsigned int *unsignedNumbers, double *uniformNumbers, double *gaussianNumbers, unsigned int totalNumbersToGenerate, unsigned int numbersToGeneratePerThread, unsigned int threadNumber);
__host__ void RNGen_Host(unsigned int numberOfBlocks, unsigned int numberOfThreadsPerBlock, RNGCombinedGenerator *generators, unsigned int numbersToGeneratePerThread, double *uniformNumbers, double *gaussianNumbers, unsigned int totalNumbersToGenerate);

int main(){
	
	unsigned int numberOfBlocks = 10;
	unsigned int numberOfThreadsPerBlock = 512;
	unsigned int totalNumberOfThreads = numberOfBlocks * numberOfThreadsPerBlock;	// Exp 6
//	unsigned int numbersToGeneratePerThread = 100;
//	unsigned int totalNumbersToGenerate = totalNumberOfThreads * numbersToGeneratePerThread;	// Exp. 24

	unsigned int totalNumbersToGenerate = 20;
	unsigned int numbersToGeneratePerThread = ceil(static_cast<double>(totalNumbersToGenerate) / totalNumberOfThreads);
	cout << "Total numbers to generate: " << totalNumbersToGenerate << endl;
	cout << "Total number of threads: " << totalNumberOfThreads << endl;
	cout << "Total numbers to generate per thread (exp. 2): " << numbersToGeneratePerThread << endl;
/*
	cout << "###############################################" << endl;
	cout << "###############################################" << endl;
	cout << "################ RNG DEBUGGING ################" << endl;
	cout << "###############################################" << endl;
	cout << "###############################################" << endl << endl;
	
	cout << "################## HOST SIDE ##################" << endl << endl << endl;
	
	cout << "################# BASIC DATA ##################" << endl << endl;
	
	cout << "Number of blocks (exp. 2): " << numberOfBlocks << endl;
	cout << "Number of threads per blocks (exp. 3): " << numberOfThreadsPerBlock << endl;
	cout << "Total number of threads (exp. 6): " << totalNumberOfThreads << endl;

	cout << "How many numbers each thread generates (exp. 4): " << numbersToGeneratePerThread << endl;
	cout << "How many numbers are generated in total (exp. 24): " << totalNumbersToGenerate << endl;
*/
	// Mersenne random generator of unsigned ints, courtesy of C++11
	mt19937 mersenneCoreGenerator(time(NULL));
	uniform_int_distribution<unsigned int> uniformDistribution(129, UINT_MAX);
	
	cout << "Maximum unsigned int, aka endpoint of uniform seed distribution: " << UINT_MAX << endl << endl;

//	cout << "################# SEED GENERATION ##################" << endl << endl;
	// Genero i seed in versione struct
	// Non sprecate tempo con i cout: ho verificato, i seed generati sono casuali e giusti
	RNGCombinedGenerator *generators = new RNGCombinedGenerator[totalNumberOfThreads];
//	cout << "Seeds during for cycle (exp. between 0 and " << UINT_MAX << ", from 0 to " << totalNumberOfThreads-1 << "): " << endl;
//	cout << "thread number\t seedLCGS\t seedTaus1\t seedTaus2\t seedTaus3" << endl;
	for(unsigned int threadNumber=0; threadNumber<totalNumberOfThreads; ++threadNumber){
		generators[threadNumber].SetInternalState(uniformDistribution(mersenneCoreGenerator),
													uniformDistribution(mersenneCoreGenerator),
													uniformDistribution(mersenneCoreGenerator),
													uniformDistribution(mersenneCoreGenerator));
		
//		cout << threadNumber
//		<< "\t" << generators[threadNumber].GetSeedLCGS()
//		<< "\t" << generators[threadNumber].GetSeedTaus1()
//		<< "\t" << generators[threadNumber].GetSeedTaus2()
//		<< "\t" << generators[threadNumber].GetSeedTaus3() << endl;
	}
//	cout << "Seeds after for cycle (exp. between 0 and " << UINT_MAX << ", from 0 to " << totalNumberOfThreads-1 << "): " << endl;
//	cout << "thread number\t seedLCGS\t seedTaus1\t seedTaus2\t seedTaus3" << endl;
//	for(unsigned int threadNumber=0; threadNumber<totalNumberOfThreads; ++threadNumber){
//		cout << threadNumber
//		<< "\t" << generators[threadNumber].GetSeedLCGS()
//		<< "\t" << generators[threadNumber].GetSeedTaus1()
//		<< "\t" << generators[threadNumber].GetSeedTaus2()
//		<< "\t" << generators[threadNumber].GetSeedTaus3() << endl;
//	}
	cout << endl;

//	cout << "Generating " << totalNumbersToGenerate << "-sized arrays for RNGs (exp. 24)" << endl;
	unsigned int *unsignedNumbers = new unsigned int[totalNumbersToGenerate];
	double *uniformNumbers = new double[totalNumbersToGenerate];
	double *gaussianNumbers = new double[totalNumbersToGenerate];

//	cout << "################# DFVICE SIDE #################" << endl << endl << endl;

/*
	////////////// HOST-SIDE GENERATOR //////////////
	RNGen_Host(numberOfBlocks, numberOfThreadsPerBlock, generators, numbersToGeneratePerThread, uniformNumbers, gaussianNumbers);
	/////////////////////////////////////////////////
*/

///*
	////////////// DEVICE-SIDE GENERATOR //////////////
	RNGCombinedGenerator *device_generators;
	unsigned int *device_unsignedNumbers;
	double *device_uniformNumbers, *device_gaussianNumbers;
	
	cudaMalloc( (void **)&device_generators, totalNumberOfThreads*sizeof(RNGCombinedGenerator) );
	cudaMalloc( (void **)&device_unsignedNumbers, totalNumbersToGenerate*sizeof(unsigned int) );
	cudaMalloc( (void **)&device_uniformNumbers, totalNumbersToGenerate*sizeof(double) );
	cudaMalloc( (void **)&device_gaussianNumbers, totalNumbersToGenerate*sizeof(double) );
	
	cudaMemcpy(device_generators, generators, totalNumberOfThreads*sizeof(RNGCombinedGenerator), cudaMemcpyHostToDevice);
	
	RNGen_Global<<<numberOfBlocks,numberOfThreadsPerBlock>>>(device_generators, device_unsignedNumbers, device_uniformNumbers, device_gaussianNumbers, totalNumbersToGenerate, numbersToGeneratePerThread);

	cudaMemcpy(unsignedNumbers, device_unsignedNumbers, totalNumbersToGenerate*sizeof(unsigned int), cudaMemcpyDeviceToHost);
	cudaMemcpy(uniformNumbers, device_uniformNumbers, totalNumbersToGenerate*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(gaussianNumbers, device_gaussianNumbers, totalNumbersToGenerate*sizeof(double), cudaMemcpyDeviceToHost);

	cudaFree(device_generators);
	cudaFree(device_unsignedNumbers);
	cudaFree(device_uniformNumbers);
	cudaFree(device_gaussianNumbers);
	///////////////////////////////////////////////////
//*/

/*	cout << "################## HOST SIDE ##################" << endl << endl << endl;
	cout << "################# SEED CHECK ##################" << endl << endl;
	cout << "Seeds after run (exp. between 0 and " << UINT_MAX << ", from 0 to 5), should be equal to last check: " << endl;
	cout << "thread number\t seedLCGS\t seedTaus1\t seedTaus2\t seedTaus3" << endl;*/
//	for(unsigned int threadNumber=0; threadNumber<totalNumberOfThreads; ++threadNumber){
//		cout << threadNumber
//		<< "\t" << generators[threadNumber].GetSeedLCGS()
//		<< "\t" << generators[threadNumber].GetSeedTaus1()
//		<< "\t" << generators[threadNumber].GetSeedTaus2()
//		<< "\t" << generators[threadNumber].GetSeedTaus3() << endl;
//	}
	cout << endl;
	
	cout << "############### OUTPUT NUMBERS ################" << endl << endl;
	cout << "RNG (exp. uniform [0,1] and gaussian (0,1), from 0 to 23): " << endl;
	cout << "thread number\t uniform\t gauss3" << endl;

	cout << "Numbers to generate: " << totalNumbersToGenerate << endl;

	for(int randomNumber=0; randomNumber<totalNumbersToGenerate; ++randomNumber){
		cout << randomNumber << "\t" << unsignedNumbers[randomNumber] << "\t" << uniformNumbers[randomNumber] << "\t" << gaussianNumbers[randomNumber] << endl;
	}

	delete[] generators;
	delete[] unsignedNumbers;
	delete[] uniformNumbers;
	delete[] gaussianNumbers;

	return 0;

}

/////////////////////////////////////////////
///////////////// FUNCTIONS /////////////////
/////////////////////////////////////////////

__global__ void RNGen_Global(RNGCombinedGenerator *generators, unsigned int *unsignedNumbers, double *uniformNumbers, double *gaussianNumbers, unsigned int totalNumbersToGenerate, unsigned int numbersToGeneratePerThread){
	unsigned int threadNumber = threadIdx.x + blockDim.x * blockIdx.x;
	RNGen_HostDev(generators, unsignedNumbers, uniformNumbers, gaussianNumbers, totalNumbersToGenerate, numbersToGeneratePerThread, threadNumber);
}

__host__ __device__ void RNGen_HostDev(RNGCombinedGenerator *generators, unsigned int *unsignedNumbers, double *uniformNumbers, double *gaussianNumbers, unsigned int totalNumbersToGenerate, unsigned int numbersToGeneratePerThread, unsigned int threadNumber){
	
	unsigned int unsignedNumber;
	double gaussian, uniform;

//	cout << "Inside thread no. (exp. equal to precedent): " << threadNumber << endl;

	for(unsigned int RNGNumber=0; RNGNumber<numbersToGeneratePerThread; ++RNGNumber){
//		cout << endl << endl << "RNGNumber (exp. between 0 and 3): " << RNGNumber << endl;
//		cout << "numbersToGeneratePerThread*threadNumber+RNGNumber (exp. between 0 and 23): " << numbersToGeneratePerThread*threadNumber+RNGNumber << endl;
//		cout << "numbersToGeneratePerThread * numberOfBlocks*numberOfThreadsPerBlock (exp. 24): " << numbersToGeneratePerThread * numberOfBlocks*numberOfThreadsPerBlock << endl;
		
		
		if(numbersToGeneratePerThread*threadNumber+RNGNumber < totalNumbersToGenerate){
//			cout << "Condizione di if soddisfatta." << endl;
			
//			cout << "*** SEED CHECK ***" << endl;
//			cout << "LCGS: " << generators[threadNumber].GetSeedLCGS() << endl;
//			cout << "Taus1: " << generators[threadNumber].GetSeedTaus1() << endl;
//			cout << "Taus2: " << generators[threadNumber].GetSeedTaus2() << endl;
//			cout << "Taus3: " << generators[threadNumber].GetSeedTaus3() << endl << endl;

			unsignedNumber = generators[threadNumber].GetUnsignedInt();
			unsignedNumbers[numbersToGeneratePerThread*threadNumber+RNGNumber] = unsignedNumber;
			
			// Ho verificato che il problema sta nell'implementazione di HybridTaus o dei suoi sottoposti
			uniform = generators[threadNumber].GetUniform();
//			cout << "Uniform: " << uniform << endl;
			uniformNumbers[numbersToGeneratePerThread*threadNumber+RNGNumber] = uniform;
			
//			cout << "*** SEED CHECK ***" << endl;
//			cout << "LCGS: " << generators[threadNumber].GetSeedLCGS() << endl;
//			cout << "Taus1: " << generators[threadNumber].GetSeedTaus1() << endl;
//			cout << "Taus2: " << generators[threadNumber].GetSeedTaus2() << endl;
//			cout << "Taus3: " << generators[threadNumber].GetSeedTaus3() << endl << endl;
			
			gaussian = generators[threadNumber].GetGauss();
//			cout << "Gaussian: " << gaussian << endl;
			gaussianNumbers[numbersToGeneratePerThread*threadNumber+RNGNumber] = gaussian;
			
//			cout << "*** SEED CHECK ***" << endl;
//			cout << "LCGS: " << generators[threadNumber].GetSeedLCGS() << endl;
//			cout << "Taus1: " << generators[threadNumber].GetSeedTaus1() << endl;
//			cout << "Taus2: " << generators[threadNumber].GetSeedTaus2() << endl;
//			cout << "Taus3: " << generators[threadNumber].GetSeedTaus3() << endl << endl;
		}
	}
}

__host__ void RNGen_Host(unsigned int numberOfBlocks, unsigned int numberOfThreadsPerBlock, RNGCombinedGenerator *generators, unsigned int *unsignedNumbers, double *uniformNumbers, double *gaussianNumbers, unsigned int totalNumbersToGenerate, unsigned int numbersToGeneratePerThread){
	
	for(unsigned int threadNumber=0; threadNumber<numberOfBlocks*numberOfThreadsPerBlock; ++threadNumber){
		cout << "Thread no. (exp. between 0 and 5): " << threadNumber << endl;
			RNGen_HostDev(generators, unsignedNumbers, uniformNumbers, gaussianNumbers, totalNumbersToGenerate, numbersToGeneratePerThread, threadNumber);
	}
}

