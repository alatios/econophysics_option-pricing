#include "Support_functions.cuh"

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
				exactPath.ExactLogNormalStep(mainGenerator->GetGauss());
				eulerPath.EulerLogNormalStep(mainGenerator->GetGauss());
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
