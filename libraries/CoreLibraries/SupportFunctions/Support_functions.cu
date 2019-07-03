#include <iostream>

#include "Support_functions.cuh"
#include "../../InputStructures/InputGPUData/Input_gpu_data.cuh"
#include "../../InputStructures/InputMarketData/Input_market_data.cuh"
#include "../../InputStructures/InputMCData/Input_MC_data.cuh"
#include "../../InputStructures/InputOptionData/Input_option_data.cuh"
#include "../DataStreamManager/Data_stream_manager.cuh"
#include "../Path/Path.cuh"
#include "../Statistics/Statistics.cuh"
#include "../RandomGenerator/RNG.cuh"
#include "../../OutputStructures/OutputMCData/Output_MC_data.cuh"

using namespace std;

// Main evaluators
__host__ __device__ void OptionPricingEvaluator_HostDev(Input_gpu_data inputGPU, Input_option_data option, Input_market_data market, Input_MC_data inputMC, Statistics* exactOutputs, Statistics* eulerOutputs, unsigned int seed, unsigned int threadNumber){
	
	unsigned int numberOfPathsPerThread = inputMC.GetNumberOfSimulationsPerThread(inputGPU);
	unsigned int numberOfIntervals = option.NumberOfIntervals;
	unsigned int totalNumberOfSimulations = inputMC.NumberOfMCSimulations;

	RNG *supportGenerator = new RNG_Tausworthe(seed+threadNumber);
	RNG *mainGenerator = new RNG_CombinedGenerator;
	mainGenerator->SetInternalState(supportGenerator);
	
	// Dummy variables to reduce memory accesses
	Path exactPath, eulerPath;

	// Cycling through paths, overwriting the same dummy path with the same template path
	for(unsigned int pathNumber=0; pathNumber<numberOfPathsPerThread; ++pathNumber){
		
		// Check if we're not overflowing. Since we decide a priori the number of simulations, some threads will inevitably work less
		if(numberOfPathsPerThread * threadNumber + pathNumber < totalNumberOfSimulations){
			exactPath.ResetToInitialState(market, option);
			eulerPath.ResetToInitialState(market, option);

			// Cycling through steps in each path
			for(unsigned int stepNumber=0; stepNumber<numberOfIntervals; ++stepNumber){
				if(inputMC.GaussianOrBimodal == 'g'){
					exactPath.ExactLogNormalStep(mainGenerator->GetGauss());
					eulerPath.EulerLogNormalStep(mainGenerator->GetGauss());
				}else if(inputMC.GaussianOrBimodal == 'b'){
					exactPath.ExactLogNormalStep(mainGenerator->GetBimodal());
					eulerPath.EulerLogNormalStep(mainGenerator->GetBimodal());
				}
			}

			exactOutputs[threadNumber].AddPayoff(exactPath.GetActualizedPayoff());
			eulerOutputs[threadNumber].AddPayoff(eulerPath.GetActualizedPayoff());
		}
	}
}

__host__ void OptionPricingEvaluator_Host(Input_gpu_data inputGPU, Input_option_data option, Input_market_data market, Input_MC_data inputMC, Statistics* exactOutputs, Statistics* eulerOutputs, unsigned int seed){
	unsigned int totalNumberOfThreads = inputGPU.GetTotalNumberOfThreads();
	
	for(unsigned int threadNumber=0; threadNumber<totalNumberOfThreads; ++threadNumber)
		OptionPricingEvaluator_HostDev(inputGPU, option, market, inputMC, exactOutputs, eulerOutputs, seed, threadNumber);
}

__global__ void OptionPricingEvaluator_Global(Input_gpu_data inputGPU, Input_option_data option, Input_market_data market, Input_MC_data inputMC, Statistics* exactOutputs, Statistics* eulerOutputs, unsigned int seed){
	unsigned int threadNumber = threadIdx.x + blockDim.x * blockIdx.x;
	OptionPricingEvaluator_HostDev(inputGPU, option, market, inputMC, exactOutputs, eulerOutputs, seed, threadNumber);
}

// CPU and GPU algorithms
__host__ void CPUOptionPricingMonteCarloAlgorithm(Data_stream_manager streamManager, Input_gpu_data inputGPU, Input_option_data inputOption, Input_market_data inputMarket, Input_MC_data inputMC, unsigned int seed){
	unsigned int numberOfThreadsPerBlock = inputGPU.GetNumberOfThreadsPerBlock();
	unsigned int totalNumberOfThreads = inputGPU.GetTotalNumberOfThreads();
	unsigned int numberOfSimulationsPerThread = inputMC.GetNumberOfSimulationsPerThread(inputGPU);

	// Time events creation
	cudaEvent_t cpuEventStart, cpuEventStop;
	float cpuElapsedTime;
	cudaEventCreate(&cpuEventStart);
	cudaEventCreate(&cpuEventStop);
	
	// Output arrays
	Statistics *cpu_exactOutputs = new Statistics[totalNumberOfThreads];
	Statistics *cpu_eulerOutputs = new Statistics[totalNumberOfThreads];
	
	cudaEventRecord(cpuEventStart,0);
	
	// Simulation of device path generation
	cout << endl << "Beginning device simulation through CPU..." << endl;
	OptionPricingEvaluator_Host(inputGPU, inputOption, inputMarket, inputMC, cpu_exactOutputs, cpu_eulerOutputs, seed);
	
	cudaEventRecord(cpuEventStop,0);
	cudaEventSynchronize(cpuEventStop);
	cudaEventElapsedTime(&cpuElapsedTime, cpuEventStart, cpuEventStop);
	
	// Output computation
	Statistics cpu_exactResults;
	Statistics cpu_eulerResults;

	for(unsigned int threadNumber=0; threadNumber<totalNumberOfThreads; ++threadNumber){
		cpu_exactResults += cpu_exactOutputs[threadNumber];
		cpu_eulerResults += cpu_eulerOutputs[threadNumber];
	}

	cpu_exactResults.EvaluateEstimatedPriceAndError();
	cpu_eulerResults.EvaluateEstimatedPriceAndError();

	// Global output MC
	Output_MC_data cpu_outputMC;
	streamManager.StoreOutputData(cpu_outputMC, cpu_exactResults, cpu_eulerResults, cpuElapsedTime, 'h');
	streamManager.PrintOutputData(cpu_outputMC);
	
	// Trash bin section, where segfaults come to die
	delete[] cpu_exactOutputs;
	delete[] cpu_eulerOutputs;
	cudaEventDestroy(cpuEventStart);
	cudaEventDestroy(cpuEventStop);	
}

__host__ void GPUOptionPricingMonteCarloAlgorithm(Data_stream_manager streamManager, Input_gpu_data inputGPU, Input_option_data inputOption, Input_market_data inputMarket, Input_MC_data inputMC, unsigned int seed){
	unsigned int numberOfThreadsPerBlock = inputGPU.GetNumberOfThreadsPerBlock();
	unsigned int totalNumberOfThreads = inputGPU.GetTotalNumberOfThreads();
	unsigned int numberOfSimulationsPerThread = inputMC.GetNumberOfSimulationsPerThread(inputGPU);

	// Time events creation
	cudaEvent_t gpuEventStart, gpuEventStop;
	float gpuElapsedTime;
	cudaEventCreate(&gpuEventStart);
	cudaEventCreate(&gpuEventStop);

	// Output arrays
	Statistics *gpu_exactOutputs = new Statistics[totalNumberOfThreads];
	Statistics *gpu_eulerOutputs = new Statistics[totalNumberOfThreads];
	
	cudaEventRecord(gpuEventStart,0);

	// Memory allocation on GPU
	Statistics *device_gpu_exactOutputs;
	Statistics *device_gpu_eulerOutputs;
	
	cudaMalloc((void **)&device_gpu_exactOutputs, totalNumberOfThreads*sizeof(Statistics));
	cudaMalloc((void **)&device_gpu_eulerOutputs, totalNumberOfThreads*sizeof(Statistics));
	
	cudaMemcpy(device_gpu_exactOutputs, gpu_exactOutputs, totalNumberOfThreads*sizeof(Statistics), cudaMemcpyHostToDevice);
	cudaMemcpy(device_gpu_eulerOutputs, gpu_eulerOutputs, totalNumberOfThreads*sizeof(Statistics), cudaMemcpyHostToDevice);
	
	// Generation of paths
	cout << endl << "Beginning GPU computation..." << endl;
	OptionPricingEvaluator_Global<<<inputGPU.NumberOfBlocks,numberOfThreadsPerBlock>>>(inputGPU, inputOption, inputMarket, inputMC, device_gpu_exactOutputs, device_gpu_eulerOutputs, seed);
	
	// The memories are coming back
	cudaMemcpy(gpu_exactOutputs, device_gpu_exactOutputs, totalNumberOfThreads*sizeof(Statistics), cudaMemcpyDeviceToHost);
	cudaMemcpy(gpu_eulerOutputs, device_gpu_eulerOutputs, totalNumberOfThreads*sizeof(Statistics), cudaMemcpyDeviceToHost);

	cudaFree(device_gpu_exactOutputs);
	cudaFree(device_gpu_eulerOutputs);
	
	cudaEventRecord(gpuEventStop,0);
	cudaEventSynchronize(gpuEventStop);
	cudaEventElapsedTime(&gpuElapsedTime, gpuEventStart, gpuEventStop);
	
	// Output computation
	Statistics gpu_exactResults;
	Statistics gpu_eulerResults;

	for(unsigned int threadNumber=0; threadNumber<totalNumberOfThreads; ++threadNumber){
		gpu_exactResults += gpu_exactOutputs[threadNumber];
		gpu_eulerResults += gpu_eulerOutputs[threadNumber];
	}

	gpu_exactResults.EvaluateEstimatedPriceAndError();
	gpu_eulerResults.EvaluateEstimatedPriceAndError();

	// Global output MC
	Output_MC_data gpu_outputMC;
	streamManager.StoreOutputData(gpu_outputMC, gpu_exactResults, gpu_eulerResults, gpuElapsedTime, 'd');
	streamManager.PrintOutputData(gpu_outputMC);
	
	// Trash bin section, where segfaults come to die
	delete[] gpu_exactOutputs;
	delete[] gpu_eulerOutputs;
	cudaEventDestroy(gpuEventStart);
	cudaEventDestroy(gpuEventStop);
}
