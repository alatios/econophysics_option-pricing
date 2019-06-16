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

__host__ __device__ void GenerateRandomNumbers_HostDev(RandomNumberGenerator **generators, unsigned int *unsignedNumbers, double *uniformNumbers, double *gaussianNumbers, unsigned int totalNumbersToGenerate, unsigned int numbersToGeneratePerThread, unsigned int threadNumber);
__host__ void GenerateRandomNumbers_Host(RandomNumberGenerator **generators, unsigned int *unsignedNumbers, double *uniformNumbers, double *gaussianNumbers, unsigned int totalNumbersToGenerate, unsigned int numbersToGeneratePerThread, unsigned int numberOfBlocks, unsigned int numberOfThreadsPerBlock);
__global__ void GenerateRandomNumbers_Global(RandomNumberGenerator **generators, unsigned int *unsignedNumbers, double *uniformNumbers, double *gaussianNumbers, unsigned int totalNumbersToGenerate, unsigned int numbersToGeneratePerThread);


int main(){
	
	unsigned int numberOfBlocks = 2;
	unsigned int numberOfThreadsPerBlock = 512;
	unsigned int totalNumberOfThreads = numberOfBlocks * numberOfThreadsPerBlock;
	
	unsigned int totalNumbersToGenerate = 20;
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

/*
	////////////// HOST-SIDE GENERATOR //////////////
	GenerateRandomNumbers_Host(generators, unsignedNumbers, uniformNumbers, gaussianNumbers, totalNumbersToGenerate, numbersToGeneratePerThread, numberOfBlocks, numberOfThreadsPerBlock);
	/////////////////////////////////////////////////
*/

///*
	////////////// DEVICE-SIDE GENERATOR //////////////
	unsigned int* device_unsignedNumbers;
	double *device_uniformNumbers, *device_gaussianNumbers;


	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	// What am I doing here? First, I declare a "support array of pointers".
	// This is analogous to generators. Its use will be clear in a moment.
	RandomNumberGenerator **host_generators;
	
	// Here, I use malloc to allocate a <totalNumberOfThreads> number of pointers within host_generators, i.e. an array of pointers
	// This is (is it?) analogous to float** ptr = new float*[20], the standard procedure for matrix allocation.
	// The difference is these pointers are not "initialized". They are not pointing to anything.
	host_generators = (RandomNumberGenerator**)malloc(totalNumberOfThreads * sizeof(RandomNumberGenerator*));
	
	// Now for the tricky part, let's start by cycling through threads
	for(unsigned int threadNumber=0; threadNumber<totalNumberOfThreads; ++threadNumber){
		// The pointers are not pointing to anything, right? Therefore it stands to reason that I can have them point to the GPU!
		// Specifically, I'm using cudaMalloc to allocate each of the pointed memory areas to a RandomNumberGenerator_Hybrid object.
		cudaMalloc( (void **)&host_generators[threadNumber], sizeof(RandomNumberGenerator_Hybrid) );
		
		// Now that I have the space, I can copy each of my previously constructed generators to this area of the GPU memory.
		// Right now, this memory is pointed to by host_generators[threadNumber], which lives in the GPU. However, THESE pointers
		// are pointed to by host_generators, which is still on CPU (hence the name).
		cudaMemcpy(host_generators[threadNumber], generators[threadNumber], sizeof(RandomNumberGenerator_Hybrid), cudaMemcpyHostToDevice);
	}
	
	// Now we're doing the big one. The goal is to have a pointer to all these pointers, and have it on GPU.
	RandomNumberGenerator **device_generators;
	
	// Standard fare: we allocate memory for totalNumberOfThreads pointers to RandomNumberGenerator objects.
	cudaMalloc( (void **)&device_generators, totalNumberOfThreads*sizeof(RandomNumberGenerator*) );
	
	// Finally, we copy the content of host_generators (totalNumberOfThreads pointers to GPU memory) to device_generators.
	// No segfault, this must mean we're doing something right.
	cudaMemcpy(device_generators, host_generators, totalNumberOfThreads*sizeof(RandomNumberGenerator*), cudaMemcpyHostToDevice);
	
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	cudaMalloc( (void **)&device_unsignedNumbers, totalNumbersToGenerate*sizeof(unsigned int) );
	cudaMalloc( (void **)&device_uniformNumbers, totalNumbersToGenerate*sizeof(double) );
	cudaMalloc( (void **)&device_gaussianNumbers, totalNumbersToGenerate*sizeof(double) );
		
	GenerateRandomNumbers_Global<<<numberOfBlocks,numberOfThreadsPerBlock>>>(device_generators, device_unsignedNumbers, device_uniformNumbers, device_gaussianNumbers, totalNumbersToGenerate, numbersToGeneratePerThread);
	
	cudaMemcpy(unsignedNumbers, device_unsignedNumbers, totalNumbersToGenerate*sizeof(unsigned int), cudaMemcpyDeviceToHost);
	cudaMemcpy(uniformNumbers, device_uniformNumbers, totalNumbersToGenerate*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(gaussianNumbers, device_gaussianNumbers, totalNumbersToGenerate*sizeof(double), cudaMemcpyDeviceToHost);

/*
	for(unsigned int threadNumber=0; threadNumber<totalNumberOfThreads; ++threadNumber){
		cudaFree(host_generators[threadNumber]);
	}
	free(host_generators);
*/
	cudaFree(device_generators);
	cudaFree(device_unsignedNumbers);
	cudaFree(device_uniformNumbers);
	cudaFree(device_gaussianNumbers);
//*/


	for(int randomNumber=0; randomNumber<totalNumbersToGenerate; ++randomNumber){
		cout << randomNumber << "\t" << unsignedNumbers[randomNumber] << "\t" << uniformNumbers[randomNumber] << "\t" << gaussianNumbers[randomNumber] << endl;
	}

	
	for(unsigned int threadNumber=0; threadNumber<totalNumberOfThreads; ++threadNumber)
		delete generators[threadNumber];
		
	delete[] generators;
	delete[] unsignedNumbers;
	delete[] uniformNumbers;
	delete[] gaussianNumbers;
	
	return 0;
}

__host__ __device__ void GenerateRandomNumbers_HostDev(RandomNumberGenerator **generators, unsigned int *unsignedNumbers, double *uniformNumbers, double *gaussianNumbers, unsigned int totalNumbersToGenerate, unsigned int numbersToGeneratePerThread, unsigned int threadNumber){
	
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
	
	/*
			unsignedNumbers[numbersToGeneratePerThread*threadNumber+RNGNumber] = 1;
			uniformNumbers[numbersToGeneratePerThread*threadNumber+RNGNumber] = 0.1;
			gaussianNumbers[numbersToGeneratePerThread*threadNumber+RNGNumber] = 0.1;
*/		}
	}

}

__host__ void GenerateRandomNumbers_Host(RandomNumberGenerator **generators, unsigned int *unsignedNumbers, double *uniformNumbers, double *gaussianNumbers, unsigned int totalNumbersToGenerate, unsigned int numbersToGeneratePerThread, unsigned int numberOfBlocks, unsigned int numberOfThreadsPerBlock){
	
	for(unsigned int threadNumber=0; threadNumber<numberOfBlocks*numberOfThreadsPerBlock; ++threadNumber){
		GenerateRandomNumbers_HostDev(generators, unsignedNumbers, uniformNumbers, gaussianNumbers, totalNumbersToGenerate, numbersToGeneratePerThread, threadNumber);
	}
}

__global__ void GenerateRandomNumbers_Global(RandomNumberGenerator **generators, unsigned int *unsignedNumbers, double *uniformNumbers, double *gaussianNumbers, unsigned int totalNumbersToGenerate, unsigned int numbersToGeneratePerThread){
	unsigned int threadNumber = threadIdx.x + blockDim.x * blockIdx.x;
	GenerateRandomNumbers_HostDev(generators, unsignedNumbers, uniformNumbers, gaussianNumbers, totalNumbersToGenerate, numbersToGeneratePerThread, threadNumber);
}


