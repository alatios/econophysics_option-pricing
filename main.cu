#include <iostream>
#include <ctime>		// time(NULL)
#include <climits>		// UINT_MAX

#include "libraries/InputStructures/InputGPUData/Input_gpu_data.cuh"
#include "libraries/InputStructures/InputMarketData/Input_market_data.cuh"
#include "libraries/InputStructures/InputMCData/Input_MC_data.cuh"
#include "libraries/InputStructures/InputOptionData/Input_option_data.cuh"
#include "libraries/CoreLibraries/DataStreamManager/Data_stream_manager.cuh"
#include "libraries/CoreLibraries/SupportFunctions/Support_functions.cuh"

using namespace std;

int main(){
	
	// Read & print input data from file
	Data_stream_manager streamManager("input.dat");
	
	Input_gpu_data inputGPU;
	Input_option_data inputOption;
	Input_market_data inputMarket;
	Input_MC_data inputMC;
	streamManager.ReadInputData(inputGPU, inputOption, inputMarket, inputMC);
	streamManager.PrintInputData(inputGPU, inputOption, inputMarket, inputMC);

	// Seed for random number generation
	// Fix it to a value between 129 and UINT_MAX-totalNumberOfThreads or let time(NULL) do its magic
	unsigned int seed;	
	do
		seed = time(NULL);
	while(seed < 129 || seed > UINT_MAX - inputGPU.GetTotalNumberOfThreads());	
	
	////////////// HOST-SIDE ALGORITHM //////////////
	if(inputMC.CpuOrGpu == 'c' || inputMC.CpuOrGpu == 'b')
		CPUOptionPricingMonteCarloAlgorithm(streamManager, inputGPU, inputOption, inputMarket, inputMC, seed);


	////////////// DEVICE-SIDE ALGORITHM //////////////
	if(inputMC.CpuOrGpu == 'g' || inputMC.CpuOrGpu == 'b')
		GPUOptionPricingMonteCarloAlgorithm(streamManager, inputGPU, inputOption, inputMarket, inputMC, seed);

	return 0;
}
