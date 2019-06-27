#include <iostream>
#include <cstdlib>
#include <fstream>		// ifstream
#include <ctime>		// time(NULL)
#include <random>		// C++11 Mersenne twister
#include <climits>		// UINT_MAX
#include <cmath>		// log, cos, sin, ceil, M_PI
#include <algorithm>	// min
#include <cstdio>
#include <tuple>		// tuple, tie, make_tuple
#include <vector>		// vector
#include <string>		// string, stoul, stod, at

#include "libraries/InputStructures/InputGPUData/Input_gpu_data.cuh"
#include "libraries/InputStructures/InputMarketData/Input_market_data.cuh"
#include "libraries/InputStructures/InputMCData/Input_MC_data.cuh"
#include "libraries/InputStructures/InputOptionData/Input_option_data.cuh"
#include "libraries/CoreLibraries/Statistics/Statistics.cuh"
#include "libraries/CoreLibraries/Path/Path.cuh"
#include "libraries/CoreLibraries/RandomGenerator/rng.cuh"
#include "libraries/CoreLibraries/SupportFunctions/Support_functions.cuh"
#include "libraries/OutputStructures/OutputMCData/Output_MC_data.cuh"

using namespace std;

int main(){
	
	vector<string> inputDataVector;
	string sourceFile = "input.dat";
	ReadInputData(inputDataVector, sourceFile);
	
	// Input GPU data
	Input_gpu_data inputGPU;
	inputGPU.NumberOfBlocks = stoul(inputDataVector[0]);
	unsigned int numberOfThreadsPerBlock = inputGPU.GetNumberOfThreadsPerBlock();
	unsigned int totalNumberOfThreads = inputGPU.GetTotalNumberOfThreads();
	
	// Input market data
	Input_market_data inputMarket;
	inputMarket.InitialPrice = stod(inputDataVector[1]);
	inputMarket.Volatility = stod(inputDataVector[2]);
	inputMarket.RiskFreeRate = stod(inputDataVector[3]);

	// Input option data
	Input_option_data inputOption;
	inputOption.TimeToMaturity = stod(inputDataVector[4]);
	inputOption.NumberOfIntervals = stoul(inputDataVector[5]);
	inputOption.OptionType = inputDataVector[6].at(0);
	inputOption.StrikePrice = stod(inputDataVector[7]);
	inputOption.B = stod(inputDataVector[8]);
	inputOption.K = stod(inputDataVector[9]);
	inputOption.N = stod(inputDataVector[10]);

	// Input Monte Carlo data
	Input_MC_data inputMC;
	inputMC.NumberOfMCSimulations = stoul(inputDataVector[11]);
	unsigned int numberOfSimulationsPerThread = inputMC.GetNumberOfSimulationsPerThread(inputGPU);
	
	// Print input data (duh)
	PrintInputData(inputGPU, inputOption, inputMarket, inputMC);
															
	// Statistics
	Statistics *exactOutputs = new Statistics[totalNumberOfThreads];
	Statistics *eulerOutputs = new Statistics[totalNumberOfThreads];

///*
	////////////// HOST-SIDE GENERATOR //////////////	
	cout << "Beginning device simulation through CPU..." << endl;
	// Simulating device function
	OptionPricingEvaluator_Host(inputGPU, inputOption, inputMarket, inputMC, exactOutputs, eulerOutputs);
	cout << endl;
	/////////////////////////////////////////////////
//*/

/*
	////////////// DEVICE-SIDE GENERATOR //////////////
	RNGCombinedGenerator *device_randomGenerators;
	Output_MC_per_thread *device_threadOutputs;
	
	cudaMalloc((void **)&device_randomGenerators, totalNumberOfThreads*sizeof(RNGCombinedGenerator));
	cudaMalloc((void **)&device_threadOutputs, totalNumberOfThreads*sizeof(Output_MC_per_thread));
	
	cudaMemcpy(device_randomGenerators, randomGenerators, totalNumberOfThreads*sizeof(RNGCombinedGenerator), cudaMemcpyHostToDevice);
	cudaMemcpy(device_threadOutputs, threadOutputs, totalNumberOfThreads*sizeof(Output_MC_per_thread), cudaMemcpyHostToDevice);

	cout << "Beginning GPU computation..." << endl;
	OptionPricingEvaluator_Global<<<numberOfBlocks,numberOfThreadsPerBlock>>>(inputGPU, inputOption, inputMarket, inputMC, pathTemplate, device_randomGenerators, device_threadOutputs);

	cudaMemcpy(threadOutputs, device_threadOutputs, totalNumberOfThreads*sizeof(Output_MC_per_thread), cudaMemcpyDeviceToHost);

	cudaFree(device_randomGenerators);
	cudaFree(device_threadOutputs);
	///////////////////////////////////////////////////
*/
	
	// Compute results
	Statistics exactResults;
	Statistics eulerResults;
	
	for(unsigned int threadNumber=0; threadNumber<totalNumberOfThreads; ++threadNumber){
		exactResults += exactOutputs[threadNumber];
		eulerResults += eulerOutputs[threadNumber];
	}
	
	exactResults.EvaluateEstimatedPriceAndError();
	eulerResults.EvaluateEstimatedPriceAndError();
	
	// Elapsed time is temporary, will be implemented later
	double elapsedTime = 0.;
	
	// Global output MC
	Output_MC_data outputMC;
	outputMC.EstimatedPriceMCExact = exactResults.GetPayoffAverage();
	outputMC.ErrorMCExact = exactResults.GetPayoffError();
	outputMC.EstimatedPriceMCEuler = eulerResults.GetPayoffAverage();
	outputMC.ErrorMCEuler = eulerResults.GetPayoffError();
	outputMC.Tick = elapsedTime;
	
	PrintOutputData(outputMC);
	cout << endl;
	
	// Trash bin section, where segfaults come to die
	delete[] exactOutputs;
	delete[] eulerOutputs;

	return 0;
}
