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

__global__ void RNGen_Global(unsigned int *unsignedNumbers, double *uniformNumbers, double *gaussianNumbers, unsigned int totalNumbersToGenerate, unsigned int numbersToGeneratePerThread);
__host__ __device__ void RNGen_HostDev(unsigned int *unsignedNumbers, double *uniformNumbers, double *gaussianNumbers, unsigned int totalNumbersToGenerate, unsigned int numbersToGeneratePerThread, unsigned int threadNumber);
__host__ void RNGen_Host(unsigned int numberOfBlocks, unsigned int numberOfThreadsPerBlock, unsigned int *unsignedNumbers, double *uniformNumbers, double *gaussianNumbers, unsigned int totalNumbersToGenerate, unsigned int numbersToGeneratePerThread);

int main(){
	
	unsigned int numberOfBlocks = 100;
	unsigned int numberOfThreadsPerBlock = 512;
	unsigned int totalNumberOfThreads = numberOfBlocks * numberOfThreadsPerBlock;
	unsigned int totalNumbersToGenerate = 512000;

	unsigned int numbersToGeneratePerThread = ceil(static_cast<double>(totalNumbersToGenerate) / totalNumberOfThreads);
	cout << "Total numbers to generate: " << totalNumbersToGenerate << endl;
	cout << "Total number of threads: " << totalNumberOfThreads << endl;
	cout << "Total numbers to generate per thread: " << numbersToGeneratePerThread << endl;

	unsigned int *unsignedNumbers = new unsigned int[totalNumbersToGenerate];
	double *uniformNumbers = new double[totalNumbersToGenerate];
	double *gaussianNumbers = new double[totalNumbersToGenerate];

/*
	////////////// HOST-SIDE GENERATOR //////////////
	RNGen_Host(numberOfBlocks, numberOfThreadsPerBlock, unsignedNumbers, uniformNumbers, gaussianNumbers, totalNumbersToGenerate, numbersToGeneratePerThread);
	/////////////////////////////////////////////////
*/

///*
	////////////// DEVICE-SIDE GENERATOR //////////////
	unsigned int *device_unsignedNumbers;
	double *device_uniformNumbers, *device_gaussianNumbers;
	
	cudaMalloc( (void **)&device_unsignedNumbers, totalNumbersToGenerate*sizeof(unsigned int) );
	cudaMalloc( (void **)&device_uniformNumbers, totalNumbersToGenerate*sizeof(double) );
	cudaMalloc( (void **)&device_gaussianNumbers, totalNumbersToGenerate*sizeof(double) );
	
	RNGen_Global<<<numberOfBlocks,numberOfThreadsPerBlock>>>(device_unsignedNumbers, device_uniformNumbers, device_gaussianNumbers, totalNumbersToGenerate, numbersToGeneratePerThread);

	cudaMemcpy(unsignedNumbers, device_unsignedNumbers, totalNumbersToGenerate*sizeof(unsigned int), cudaMemcpyDeviceToHost);
	cudaMemcpy(uniformNumbers, device_uniformNumbers, totalNumbersToGenerate*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(gaussianNumbers, device_gaussianNumbers, totalNumbersToGenerate*sizeof(double), cudaMemcpyDeviceToHost);

	cudaFree(device_unsignedNumbers);
	cudaFree(device_uniformNumbers);
	cudaFree(device_gaussianNumbers);
	///////////////////////////////////////////////////
//*/

	cout << "############### OUTPUT NUMBERS ################" << endl << endl;
	cout << "rngnumber\t unsigned\t uniform\t gauss" << endl;

	cout << "Numbers to generate: " << totalNumbersToGenerate << endl;

	for(int randomNumber=0; randomNumber<totalNumbersToGenerate; ++randomNumber)
		cout << randomNumber << "\t" << unsignedNumbers[randomNumber] << "\t" << uniformNumbers[randomNumber] << "\t" << gaussianNumbers[randomNumber] << endl;


	delete[] unsignedNumbers;
	delete[] uniformNumbers;
	delete[] gaussianNumbers;

	return 0;

}

/////////////////////////////////////////////
///////////////// FUNCTIONS /////////////////
/////////////////////////////////////////////

__global__ void RNGen_Global(unsigned int *unsignedNumbers, double *uniformNumbers, double *gaussianNumbers, unsigned int totalNumbersToGenerate, unsigned int numbersToGeneratePerThread){
	unsigned int threadNumber = threadIdx.x + blockDim.x * blockIdx.x;
	RNGen_HostDev(unsignedNumbers, uniformNumbers, gaussianNumbers, totalNumbersToGenerate, numbersToGeneratePerThread, threadNumber);
}

__host__ __device__ void RNGen_HostDev(unsigned int *unsignedNumbers, double *uniformNumbers, double *gaussianNumbers, unsigned int totalNumbersToGenerate, unsigned int numbersToGeneratePerThread, unsigned int threadNumber){
	
	RNG *supportGenerator = new RNG_CurandAdapter(clock64(), threadNumber);
	
	RNG *mainGenerator = new RNG_CombinedGenerator;
	mainGenerator->SetInternalState(supportGenerator);
	
	unsigned int unsignedNumber;
	double gaussian, uniform;

	for(unsigned int RNGNumber=0; RNGNumber<numbersToGeneratePerThread; ++RNGNumber){		
		if(numbersToGeneratePerThread*threadNumber+RNGNumber < totalNumbersToGenerate){
			unsignedNumber = mainGenerator->GetUnsignedInt();
			unsignedNumbers[numbersToGeneratePerThread*threadNumber+RNGNumber] = unsignedNumber;

			uniform = mainGenerator->GetUniform();
			uniformNumbers[numbersToGeneratePerThread*threadNumber+RNGNumber] = uniform;
		
			gaussian = mainGenerator->GetGauss();
			gaussianNumbers[numbersToGeneratePerThread*threadNumber+RNGNumber] = gaussian;
		}
	}
}

__host__ void RNGen_Host(unsigned int numberOfBlocks, unsigned int numberOfThreadsPerBlock, unsigned int *unsignedNumbers, double *uniformNumbers, double *gaussianNumbers, unsigned int totalNumbersToGenerate, unsigned int numbersToGeneratePerThread){
	
	for(unsigned int threadNumber=0; threadNumber<numberOfBlocks*numberOfThreadsPerBlock; ++threadNumber)
			RNGen_HostDev(unsignedNumbers, uniformNumbers, gaussianNumbers, totalNumbersToGenerate, numbersToGeneratePerThread, threadNumber);
}

