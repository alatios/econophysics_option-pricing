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
	unsigned int totalNumberOfThreads = numberOfBlocks * numberOfThreadsPerBlock;
	unsigned int totalNumbersToGenerate = 5120000;

	unsigned int numbersToGeneratePerThread = ceil(static_cast<double>(totalNumbersToGenerate) / totalNumberOfThreads);
	cout << "Total numbers to generate: " << totalNumbersToGenerate << endl;
	cout << "Total number of threads: " << totalNumberOfThreads << endl;
	cout << "Total numbers to generate per thread (exp. 2): " << numbersToGeneratePerThread << endl;

	// Mersenne random generator of unsigned ints, courtesy of C++11
	mt19937 mersenneCoreGenerator(time(NULL));
	uniform_int_distribution<unsigned int> uniformDistribution(129, UINT_MAX);
	
	cout << "Maximum unsigned int, aka endpoint of uniform seed distribution: " << UINT_MAX << endl << endl;

	RNGCombinedGenerator *generators = new RNGCombinedGenerator[totalNumberOfThreads];
	for(unsigned int threadNumber=0; threadNumber<totalNumberOfThreads; ++threadNumber){
		generators[threadNumber].SetInternalState(uniformDistribution(mersenneCoreGenerator),
													uniformDistribution(mersenneCoreGenerator),
													uniformDistribution(mersenneCoreGenerator),
													uniformDistribution(mersenneCoreGenerator));
	}

	unsigned int *unsignedNumbers = new unsigned int[totalNumbersToGenerate];
	double *uniformNumbers = new double[totalNumbersToGenerate];
	double *gaussianNumbers = new double[totalNumbersToGenerate];

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
/*
	cout << "############### OUTPUT NUMBERS ################" << endl << endl;
	cout << "RNG (exp. uniform [0,1] and gaussian (0,1), from 0 to 23): " << endl;
	cout << "thread number\t uniform\t gauss3" << endl;

	cout << "Numbers to generate: " << totalNumbersToGenerate << endl;

	for(int randomNumber=0; randomNumber<totalNumbersToGenerate; ++randomNumber)
		cout << randomNumber << "\t" << unsignedNumbers[randomNumber] << "\t" << uniformNumbers[randomNumber] << "\t" << gaussianNumbers[randomNumber] << endl;
*/
	cout << endl <<  "############### INTER-STREAM TEST ###############" << endl << endl;
	int step = 0;					//To choose which step of the random generation you want
	double uniform_sum = 0;
	double uniform_sum2 = 0;
	double gaussian_sum = 0;
	double gaussian_sum2 = 0;
	for(int randomNumber=0; randomNumber<totalNumberOfThreads - 1; ++randomNumber){
		uniform_sum =+ uniformNumbers[randomNumber * numbersToGeneratePerThread + step] * uniformNumbers[(randomNumber + 1) * numbersToGeneratePerThread + step];
		gaussian_sum =+ gaussianNumbers[randomNumber * numbersToGeneratePerThread + step] * gaussianNumbers[(randomNumber + 1) * numbersToGeneratePerThread + step];
		uniform_sum2 =+ pow(uniformNumbers[randomNumber * numbersToGeneratePerThread + step],2) * pow(uniformNumbers[(randomNumber + 1) * numbersToGeneratePerThread + step ],2);
		gaussian_sum2 =+ pow(gaussianNumbers[randomNumber * numbersToGeneratePerThread + step],2) * pow(gaussianNumbers[(randomNumber + 1) * numbersToGeneratePerThread + step ],2);
	}

	double uniform_mean = uniform_sum/totalNumberOfThreads;
	double uniform_standard_deviation = sqrt(uniform_sum2/totalNumberOfThreads - pow(uniform_mean,2));
	double gaussian_mean = gaussian_sum/totalNumberOfThreads;
	double gaussian_standard_deviation = sqrt(gaussian_sum2/totalNumberOfThreads - pow(gaussian_sum/totalNumberOfThreads,2));
	cout << "Correlation function of uniform numbers: " << uniform_mean << " +- " << uniform_standard_deviation << endl;
	cout << "The value obtained differs from the expected by: " << abs(uniform_mean/uniform_standard_deviation) << " stardard deviation" << endl << endl;
	cout << "Correlation function of gaussian numbers: " << gaussian_mean << " +- " << gaussian_standard_deviation << endl;
	cout << "The value obtained differs from the expected by: " << abs(gaussian_mean/gaussian_standard_deviation) << " stardard deviation" << endl << endl;

	cout << endl <<  "############### INTRA-STREAM TEST ###############" << endl << endl;

	int thread_test = 0;			//To choose on which thread make tests
	cout << "The tests is made only on thread stream no. "  << thread_test << endl;
	uniform_sum = 0;
	double uniform_sum_corr = 0.;
	gaussian_sum = 0;
	gaussian_sum2 = 0;
	for(int randomNumber=0; randomNumber<numbersToGeneratePerThread; ++randomNumber){
		uniform_sum += uniformNumbers[randomNumber + thread_test]; 
		gaussian_sum += gaussianNumbers[randomNumber + thread_test];
		gaussian_sum2 += pow(gaussianNumbers[randomNumber + thread_test],2);
	}
	for(int randomNumber=0; randomNumber<numbersToGeneratePerThread - 1; ++randomNumber){
		uniform_sum_corr += uniformNumbers[randomNumber + thread_test] * uniformNumbers[randomNumber + 1 + thread_test];
	}


	uniform_mean = uniform_sum/numbersToGeneratePerThread;
	double uniform_correlation = uniform_sum_corr/numbersToGeneratePerThread;
	gaussian_mean = gaussian_sum/numbersToGeneratePerThread;
	gaussian_standard_deviation = sqrt(gaussian_sum2/numbersToGeneratePerThread - pow(gaussian_sum/totalNumberOfThreads,2));
	cout << "Uniform mean: " << uniform_mean << "| Uniform Correlation intrastram: " << uniform_correlation << endl;
	cout << "Gaussian mean: " << gaussian_mean << "| Gaussian standard deviation: " << gaussian_standard_deviation << endl << endl;

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

	for(unsigned int RNGNumber=0; RNGNumber<numbersToGeneratePerThread; ++RNGNumber){		
		if(numbersToGeneratePerThread*threadNumber+RNGNumber < totalNumbersToGenerate){
			unsignedNumber = generators[threadNumber].GetUnsignedInt();
			unsignedNumbers[numbersToGeneratePerThread*threadNumber+RNGNumber] = unsignedNumber;

			uniform = generators[threadNumber].GetUniform();
			uniformNumbers[numbersToGeneratePerThread*threadNumber+RNGNumber] = uniform;
		
			gaussian = generators[threadNumber].GetGauss();
			gaussianNumbers[numbersToGeneratePerThread*threadNumber+RNGNumber] = gaussian;
		}
	}
}

__host__ void RNGen_Host(unsigned int numberOfBlocks, unsigned int numberOfThreadsPerBlock, RNGCombinedGenerator *generators, unsigned int *unsignedNumbers, double *uniformNumbers, double *gaussianNumbers, unsigned int totalNumbersToGenerate, unsigned int numbersToGeneratePerThread){
	
	for(unsigned int threadNumber=0; threadNumber<numberOfBlocks*numberOfThreadsPerBlock; ++threadNumber){
		cout << "Thread no. (exp. between 0 and 5): " << threadNumber << endl;
			RNGen_HostDev(generators, unsignedNumbers, uniformNumbers, gaussianNumbers, totalNumbersToGenerate, numbersToGeneratePerThread, threadNumber);
	}
}

