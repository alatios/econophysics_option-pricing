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
#include "correlationtest.cuh"

using namespace std;

__global__ void RNGen_Global(unsigned int *unsignedNumbers, double *uniformNumbers, double *gaussianNumbers, unsigned int totalNumbersToGenerate, unsigned int numbersToGeneratePerThread);
__host__ __device__ void RNGen_HostDev(unsigned int *unsignedNumbers, double *uniformNumbers, double *gaussianNumbers, unsigned int totalNumbersToGenerate, unsigned int numbersToGeneratePerThread, unsigned int threadNumber);
__host__ void RNGen_Host(unsigned int numberOfBlocks, unsigned int numberOfThreadsPerBlock, unsigned int *unsignedNumbers, double *uniformNumbers, double *gaussianNumbers, unsigned int totalNumbersToGenerate, unsigned int numbersToGeneratePerThread);

int main(){
	
	unsigned int numberOfBlocks = 100;
	unsigned int numberOfThreadsPerBlock = 512;
	unsigned int totalNumberOfThreads = numberOfBlocks * numberOfThreadsPerBlock;
	unsigned int totalNumbersToGenerate = 51200000;

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

	correlationTest(totalNumberOfThreads, numbersToGeneratePerThread, uniformNumbers, gaussianNumbers);

/*

	cout << "############### OUTPUT NUMBERS ################" << endl << endl;
	cout << "thread number\t uniform\t gauss" << endl;

	cout << "Numbers to generate: " << totalNumbersToGenerate << endl;

	for(int randomNumber=0; randomNumber<totalNumbersToGenerate; ++randomNumber)
		cout << randomNumber << "\t" << unsignedNumbers[randomNumber] << "\t" << uniformNumbers[randomNumber] << "\t" << gaussianNumbers[randomNumber] << endl;

*/

/***************************************
	cout << endl <<  "############### INTER-STREAM TEST ###############" << endl << endl;
	int streamStep = 0;					// To choose which step of the random generation you want
	double sumOfUniformNumbers = 0.;
	double squaredSumOfUniformNumbers = 0.;
	double sumOfGaussianNumbers = 0.;
	double squaredSumOfGaussianNumbers = 0.;

	for(unsigned int randomStream=0; randomStream<totalNumberOfThreads - 1; ++randomStream){
		sumOfUniformNumbers += uniformNumbers[randomStream * numbersToGeneratePerThread + streamStep]
								* uniformNumbers[(randomStream + 1) * numbersToGeneratePerThread + streamStep];
		sumOfGaussianNumbers += gaussianNumbers[randomStream * numbersToGeneratePerThread + streamStep]
								* gaussianNumbers[(randomStream + 1) * numbersToGeneratePerThread + streamStep];
		squaredSumOfUniformNumbers += pow(uniformNumbers[randomStream * numbersToGeneratePerThread + streamStep],2)
								* pow(uniformNumbers[(randomStream + 1) * numbersToGeneratePerThread + streamStep],2);
		squaredSumOfGaussianNumbers += pow(gaussianNumbers[randomStream * numbersToGeneratePerThread + streamStep],2)
								* pow(gaussianNumbers[(randomStream + 1) * numbersToGeneratePerThread + streamStep],2);
	}

	double averageOfUniformNumbers = sumOfUniformNumbers/totalNumberOfThreads;
	double standardDeviationOfUniformNumbers = sqrt(squaredSumOfUniformNumbers/totalNumberOfThreads - pow(averageOfUniformNumbers,2));
	double averageOfGaussianNumbers = sumOfGaussianNumbers/totalNumberOfThreads;
	double standardDeviationOfGaussianNumbers = sqrt(squaredSumOfGaussianNumbers/totalNumberOfThreads - pow(averageOfGaussianNumbers,2));
	cout << "Correlation function of uniform numbers: " << averageOfUniformNumbers << " +- " << standardDeviationOfUniformNumbers << endl;
	cout << "The value obtained differs from the expected by: " << abs(averageOfUniformNumbers/standardDeviationOfUniformNumbers) << " stardard deviation" << endl << endl;
	cout << "Correlation function of gaussian numbers: " << averageOfGaussianNumbers << " +- " << standardDeviationOfGaussianNumbers << endl;
	cout << "The value obtained differs from the expected by: " << abs(averageOfGaussianNumbers/standardDeviationOfGaussianNumbers) << " stardard deviation" << endl << endl;

	cout << endl <<  "############### INTRA-STREAM TEST ###############" << endl << endl;

	int threadToTest = 0;			// To choose on which thread make tests
	cout << "The tests is made only on thread stream no. "  << threadToTest << endl;
	sumOfUniformNumbers = 0;
	squaredSumOfUniformNumbers = 0;
	double sumOfUniformNumberProducts = 0.;
	double sumOfGaussianNumberProducts = 0.;
	double squaredSumOfUniformNumberProducts = 0.;
	double squaredSumOfGaussianNumberProducts = 0.;
	sumOfGaussianNumbers = 0;
	squaredSumOfGaussianNumbers = 0;
	for(unsigned int randomNumber=0; randomNumber<numbersToGeneratePerThread; ++randomNumber){
		sumOfUniformNumbers += uniformNumbers[randomNumber + threadToTest * numbersToGeneratePerThread]; 
		sumOfGaussianNumbers += gaussianNumbers[randomNumber + threadToTest * numbersToGeneratePerThread];
		squaredSumOfGaussianNumbers += pow(gaussianNumbers[randomNumber + threadToTest * numbersToGeneratePerThread],2);
		squaredSumOfUniformNumbers += pow(uniformNumbers[randomNumber + threadToTest * numbersToGeneratePerThread], 2);
	}
	
	for(unsigned int randomNumber=0; randomNumber<numbersToGeneratePerThread - 1; ++randomNumber){
		sumOfUniformNumberProducts += uniformNumbers[randomNumber + threadToTest * numbersToGeneratePerThread] * uniformNumbers[randomNumber + 1 + threadToTest * numbersToGeneratePerThread];
		sumOfGaussianNumberProducts += gaussianNumbers[randomNumber + threadToTest * numbersToGeneratePerThread] * gaussianNumbers[randomNumber + 1 + threadToTest * numbersToGeneratePerThread];
		squaredSumOfUniformNumberProducts += pow(uniformNumbers[randomNumber + threadToTest * numbersToGeneratePerThread], 2) * pow(uniformNumbers[randomNumber + 1 + threadToTest * numbersToGeneratePerThread], 2);
		squaredSumOfGaussianNumberProducts += pow(gaussianNumbers[randomNumber + threadToTest * numbersToGeneratePerThread], 2) * pow(gaussianNumbers[randomNumber + 1 + threadToTest * numbersToGeneratePerThread], 2);
	}

	averageOfUniformNumbers = sumOfUniformNumbers/numbersToGeneratePerThread;
	standardDeviationOfUniformNumbers = sqrt(squaredSumOfUniformNumbers/numbersToGeneratePerThread - pow(averageOfUniformNumbers,2));
	double correlationOfUniformNumbers = sumOfUniformNumberProducts/numbersToGeneratePerThread;
	averageOfGaussianNumbers = sumOfGaussianNumbers/numbersToGeneratePerThread;
	standardDeviationOfGaussianNumbers = sqrt(squaredSumOfGaussianNumbers/numbersToGeneratePerThread - pow(averageOfGaussianNumbers,2));
	cout << "Uniform mean: " << averageOfUniformNumbers << "| Uniform standard deviation: " << standardDeviationOfUniformNumbers << "| Uniform Correlation intrastram: " << correlationOfUniformNumbers << endl;
	cout << "Gaussian mean: " << averageOfGaussianNumbers << "| Gaussian standard deviation: " << standardDeviationOfGaussianNumbers << endl << endl;

//Non so che formula abbia usato Riccardo qui sopra e con che criterio il test sul singolo thread sia stato fatto diverso
//da quello su tutti i thread quindi ho riportato il procedimento precedente anche in questo test intrastream perchè la 
//correlazione tra i numeri casuali sul singolo thread e tra quelli i-esimi di ogni thread dovrebbe essere la stessa credo
//Se pensate che sia sbagliato ho lasciato inalterati i cout sopra, così al massimo basta cancellare le righe in eccesso

	cout << "Correlation function of uniform numbers: " << averageOfUniformNumbers << " +- " << standardDeviationOfUniformNumbers << endl;
	cout << "The value obtained differs from the expected by: " << abs(averageOfUniformNumbers/standardDeviationOfUniformNumbers) << " stardard deviation" << endl << endl;
	cout << "Correlation function of gaussian numbers: " << averageOfGaussianNumbers << " +- " << standardDeviationOfGaussianNumbers << endl;
	cout << "The value obtained differs from the expected by: " << abs(averageOfGaussianNumbers/standardDeviationOfGaussianNumbers) << " stardard deviation" << endl << endl;

*/

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
	
	RNG *generator = new RNG_CombinedGenerator;
	generator->SetInternalState(11+threadNumber,1129+threadNumber,1130+threadNumber,1131+threadNumber);
	
	unsigned int unsignedNumber;
	double gaussian, uniform;

	for(unsigned int RNGNumber=0; RNGNumber<numbersToGeneratePerThread; ++RNGNumber){		
		if(numbersToGeneratePerThread*threadNumber+RNGNumber < totalNumbersToGenerate){
			unsignedNumber = generator->GetUnsignedInt();
			unsignedNumbers[numbersToGeneratePerThread*threadNumber+RNGNumber] = unsignedNumber;

			uniform = generator->GetUniform();
			uniformNumbers[numbersToGeneratePerThread*threadNumber+RNGNumber] = uniform;
		
			gaussian = generator->GetGauss();
			gaussianNumbers[numbersToGeneratePerThread*threadNumber+RNGNumber] = gaussian;
		}
	}
}

__host__ void RNGen_Host(unsigned int numberOfBlocks, unsigned int numberOfThreadsPerBlock, unsigned int *unsignedNumbers, double *uniformNumbers, double *gaussianNumbers, unsigned int totalNumbersToGenerate, unsigned int numbersToGeneratePerThread){
	
	for(unsigned int threadNumber=0; threadNumber<numberOfBlocks*numberOfThreadsPerBlock; ++threadNumber)
			RNGen_HostDev(unsignedNumbers, uniformNumbers, gaussianNumbers, totalNumbersToGenerate, numbersToGeneratePerThread, threadNumber);
}

