//
//	OUTPUT & CPU/GPU COMPARISON TEST
//
//	Tests correct generation of pseudorandom numbers and compares CPU to GPU output with same seed.
//

#include <iostream>
#include <ctime>		// time(NULL) for seed
#include <climits>		// UINT_MAX
#include <cmath>		// ceil

#include "RNG.cuh"

using namespace std;

__global__ void RNGen_Global(unsigned int *unsignedNumbers, double *uniformNumbers, double *gaussianNumbers, double *bimodalNumbers, unsigned int totalNumbersToGenerate, unsigned int numbersToGeneratePerThread, unsigned int seed);
__host__ __device__ void RNGen_HostDev(unsigned int *unsignedNumbers, double *uniformNumbers, double *gaussianNumbers, double *bimodalNumbers, unsigned int totalNumbersToGenerate, unsigned int numbersToGeneratePerThread, unsigned int seed, unsigned int threadNumber);
__host__ void RNGen_Host(unsigned int numberOfBlocks, unsigned int numberOfThreadsPerBlock, unsigned int *unsignedNumbers, double *uniformNumbers, double *gaussianNumbers, double *bimodalNumbers, unsigned int totalNumbersToGenerate, unsigned int numbersToGeneratePerThread, unsigned int seed);

bool AreSame(unsigned int, unsigned int);
bool AreSame(double, double);

int main(){

	unsigned int numberOfBlocks = 10;
	unsigned int numberOfThreadsPerBlock = 512;
	unsigned int totalNumberOfThreads = numberOfBlocks * numberOfThreadsPerBlock;
	unsigned int totalNumbersToGenerate = 500000;
	
	unsigned int seed;	
	
	do
		seed = time(NULL);
	while(seed < 129 || seed > UINT_MAX - totalNumberOfThreads);

	unsigned int numbersToGeneratePerThread = ceil(static_cast<double>(totalNumbersToGenerate) / totalNumberOfThreads);
	cout << "Total numbers to generate: " << totalNumbersToGenerate << endl;
	cout << "Total number of threads: " << totalNumberOfThreads << endl;
	cout << "Total numbers to generate per thread: " << numbersToGeneratePerThread << endl;

	// CPU-side results
	unsigned int *cpu_unsignedNumbers = new unsigned int[totalNumbersToGenerate];
	double *cpu_uniformNumbers = new double[totalNumbersToGenerate];
	double *cpu_gaussianNumbers = new double[totalNumbersToGenerate];
	double *cpu_bimodalNumbers = new double[totalNumbersToGenerate];

	// GPU-side results
	unsigned int *gpu_unsignedNumbers = new unsigned int[totalNumbersToGenerate];
	double *gpu_uniformNumbers = new double[totalNumbersToGenerate];
	double *gpu_gaussianNumbers = new double[totalNumbersToGenerate];
	double *gpu_bimodalNumbers = new double[totalNumbersToGenerate];
	
	////////////// HOST-SIDE GENERATOR //////////////
	RNGen_Host(numberOfBlocks, numberOfThreadsPerBlock, cpu_unsignedNumbers, cpu_uniformNumbers, cpu_gaussianNumbers, cpu_bimodalNumbers, totalNumbersToGenerate, numbersToGeneratePerThread, seed);
	/////////////////////////////////////////////////

	////////////// DEVICE-SIDE GENERATOR //////////////
	unsigned int *dev_gpu_unsignedNumbers;
	double *dev_gpu_uniformNumbers, *dev_gpu_gaussianNumbers, *dev_gpu_bimodalNumbers;
	
	cudaMalloc( (void **)&dev_gpu_unsignedNumbers, totalNumbersToGenerate*sizeof(unsigned int) );
	cudaMalloc( (void **)&dev_gpu_uniformNumbers, totalNumbersToGenerate*sizeof(double) );
	cudaMalloc( (void **)&dev_gpu_gaussianNumbers, totalNumbersToGenerate*sizeof(double) );
	cudaMalloc( (void **)&dev_gpu_bimodalNumbers, totalNumbersToGenerate*sizeof(double) );
	
	RNGen_Global<<<numberOfBlocks,numberOfThreadsPerBlock>>>(dev_gpu_unsignedNumbers, dev_gpu_uniformNumbers, dev_gpu_gaussianNumbers, dev_gpu_bimodalNumbers, totalNumbersToGenerate, numbersToGeneratePerThread, seed);

	cudaMemcpy(gpu_unsignedNumbers, dev_gpu_unsignedNumbers, totalNumbersToGenerate*sizeof(unsigned int), cudaMemcpyDeviceToHost);
	cudaMemcpy(gpu_uniformNumbers, dev_gpu_uniformNumbers, totalNumbersToGenerate*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(gpu_gaussianNumbers, dev_gpu_gaussianNumbers, totalNumbersToGenerate*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(gpu_bimodalNumbers, dev_gpu_bimodalNumbers, totalNumbersToGenerate*sizeof(double), cudaMemcpyDeviceToHost);

	cudaFree(dev_gpu_unsignedNumbers);
	cudaFree(dev_gpu_uniformNumbers);
	cudaFree(dev_gpu_gaussianNumbers);
	cudaFree(dev_gpu_bimodalNumbers);
	///////////////////////////////////////////////////
	
	////////////// TESTS //////////////

	cout << endl << "############### OUTPUT NUMBERS ################" << endl;
	
	cout << endl << "CPU: " << endl;
	cout << "thread\t unsigned\t uniform\t gauss\t bimodal" << endl;
	for(int randomNumber=0; randomNumber<5; ++randomNumber)
		cout << randomNumber << "\t" << cpu_unsignedNumbers[randomNumber] << "\t" << cpu_uniformNumbers[randomNumber] << "\t" << cpu_gaussianNumbers[randomNumber] << "\t" << cpu_bimodalNumbers[randomNumber] << endl;;
	cout << ". . ." << endl;	
	for(int randomNumber=totalNumbersToGenerate-5; randomNumber<totalNumbersToGenerate; ++randomNumber)
		cout << randomNumber << "\t" << cpu_unsignedNumbers[randomNumber] << "\t" << cpu_uniformNumbers[randomNumber] << "\t" << cpu_gaussianNumbers[randomNumber] << "\t" << cpu_bimodalNumbers[randomNumber] << endl;
		
	cout << endl << "GPU: " << endl;
	cout << "thread\t unsigned\t uniform\t gauss" << endl;
	for(int randomNumber=0; randomNumber<5; ++randomNumber)
		cout << randomNumber << "\t" << gpu_unsignedNumbers[randomNumber] << "\t" << gpu_uniformNumbers[randomNumber] << "\t" << gpu_gaussianNumbers[randomNumber] << "\t" << gpu_bimodalNumbers[randomNumber] << endl;
	cout << ". . ." << endl;	
	for(int randomNumber=totalNumbersToGenerate-5; randomNumber<totalNumbersToGenerate; ++randomNumber)
		cout << randomNumber << "\t" << gpu_unsignedNumbers[randomNumber] << "\t" << gpu_uniformNumbers[randomNumber] << "\t" << gpu_gaussianNumbers[randomNumber] << "\t" << gpu_bimodalNumbers[randomNumber] << endl;

	cout << endl << "############### GPU-CPU COMPARISON ################" << endl << endl;
	
	bool gpuCpuComparison = true;
	for(int randomNumber=0; randomNumber<totalNumbersToGenerate; ++randomNumber){
		if(!AreSame(gpu_unsignedNumbers[randomNumber], cpu_unsignedNumbers[randomNumber])){
			gpuCpuComparison = false;
			cout << "FAILED@step " << randomNumber << ":\t" << gpu_unsignedNumbers[randomNumber] << "\t" << cpu_unsignedNumbers[randomNumber] << endl;
		}
		
		if(!AreSame(gpu_uniformNumbers[randomNumber], cpu_uniformNumbers[randomNumber])){
			gpuCpuComparison = false;
			cout << "FAILED@step " << randomNumber << ":\t" << gpu_uniformNumbers[randomNumber] << "\t" << cpu_uniformNumbers[randomNumber] << endl;
		}
		
		if(!AreSame(gpu_gaussianNumbers[randomNumber], cpu_gaussianNumbers[randomNumber])){
			gpuCpuComparison = false;
			cout << "FAILED@step " << randomNumber << ":\t" << gpu_gaussianNumbers[randomNumber] << "\t" << cpu_gaussianNumbers[randomNumber] << endl;
		}
		
		if(!AreSame(gpu_bimodalNumbers[randomNumber], cpu_bimodalNumbers[randomNumber])){
			gpuCpuComparison = false;
			cout << "FAILED@step " << randomNumber << ":\t" << gpu_bimodalNumbers[randomNumber] << "\t" << cpu_bimodalNumbers[randomNumber] << endl;
		}
	}
	
	if(gpuCpuComparison)
		cout << "Test PASSED!" << endl;
	else
		cout << "Test failed..." << endl;

	delete[] cpu_unsignedNumbers;
	delete[] cpu_uniformNumbers;
	delete[] cpu_gaussianNumbers;
	delete[] cpu_bimodalNumbers;
	
	delete[] gpu_unsignedNumbers;
	delete[] gpu_uniformNumbers;
	delete[] gpu_gaussianNumbers;
	delete[] gpu_bimodalNumbers;

	return 0;

}

/////////////////////////////////////////////
///////////////// FUNCTIONS /////////////////
/////////////////////////////////////////////

__global__ void RNGen_Global(unsigned int *unsignedNumbers, double *uniformNumbers, double *gaussianNumbers, double *bimodalNumbers, unsigned int totalNumbersToGenerate, unsigned int numbersToGeneratePerThread, unsigned int seed){
	unsigned int threadNumber = threadIdx.x + blockDim.x * blockIdx.x;
	RNGen_HostDev(unsignedNumbers, uniformNumbers, gaussianNumbers, bimodalNumbers, totalNumbersToGenerate, numbersToGeneratePerThread, seed, threadNumber);
}

__host__ __device__ void RNGen_HostDev(unsigned int *unsignedNumbers, double *uniformNumbers, double *gaussianNumbers, double *bimodalNumbers, unsigned int totalNumbersToGenerate, unsigned int numbersToGeneratePerThread, unsigned int seed, unsigned int threadNumber){
	
	RNG *supportGenerator = new RNG_Tausworthe(seed+threadNumber);
	
	RNG *generator = new RNG_CombinedGenerator;
	generator->SetInternalState(supportGenerator);
	
	unsigned int unsignedNumber;
	double gaussian, uniform, bimodal;

	for(unsigned int RNGNumber=0; RNGNumber<numbersToGeneratePerThread; ++RNGNumber){		
		if(numbersToGeneratePerThread*threadNumber+RNGNumber < totalNumbersToGenerate){
			unsignedNumber = generator->GetUnsignedInt();
			unsignedNumbers[numbersToGeneratePerThread*threadNumber+RNGNumber] = unsignedNumber;

			uniform = generator->GetUniform();
			uniformNumbers[numbersToGeneratePerThread*threadNumber+RNGNumber] = uniform;
		
			gaussian = generator->GetGauss();
			gaussianNumbers[numbersToGeneratePerThread*threadNumber+RNGNumber] = gaussian;
			
			bimodal = generator->GetBimodal();
			bimodalNumbers[numbersToGeneratePerThread*threadNumber+RNGNumber] = bimodal;
		}
	}
}

__host__ void RNGen_Host(unsigned int numberOfBlocks, unsigned int numberOfThreadsPerBlock, unsigned int *unsignedNumbers, double *uniformNumbers, double *gaussianNumbers, double *bimodalNumbers, unsigned int totalNumbersToGenerate, unsigned int numbersToGeneratePerThread, unsigned int seed){
	
	for(unsigned int threadNumber=0; threadNumber<numberOfBlocks*numberOfThreadsPerBlock; ++threadNumber)
			RNGen_HostDev(unsignedNumbers, uniformNumbers, gaussianNumbers, bimodalNumbers, totalNumbersToGenerate, numbersToGeneratePerThread, seed, threadNumber);
}

bool AreSame(unsigned int a, unsigned int b){
	unsigned int diff = a - b;
	double epsilon = 0.0001;	// 0.01% difference
	return (fabs(static_cast<double>(diff) / a) < epsilon);
}
bool AreSame(double a, double b){
	double diff = a - b;
	double epsilon = 0.0001;	// 0.01% difference
	return (fabs(diff / a) < epsilon);	
}

