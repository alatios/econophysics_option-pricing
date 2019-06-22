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

#include "libraries/InputGPUData/Input_gpu_data.cuh"
#include "libraries/InputMarketData/Input_market_data.cuh"
#include "libraries/InputMCData/Input_MC_data.cuh"
#include "libraries/InputOptionData/Input_option_data.cuh"
#include "libraries/OutputMCData/Output_MC_data.cuh"
#include "libraries/Path/Path.cuh"
#include "libraries/Statistics/Statistics.cuh"
#include "random_generator/rng.cuh"
#include "general_purpose_functions/Support_functions.cuh"

using namespace std;

int main(){
	
	vector<string> inputDataVector;
	string sourceFile = "input.dat";
	ReadInputData(inputDataVector, sourceFile);
	
	// Input GPU data
	unsigned int numberOfBlocks = stoul(inputDataVector[0]);
	Input_gpu_data inputGPU(numberOfBlocks);
	unsigned int numberOfThreadsPerBlock = inputGPU.GetNumberOfThreadsPerBlock();
	unsigned int totalNumberOfThreads = inputGPU.GetTotalNumberOfThreads();
	
	// Input market data
	double initialPrice = stod(inputDataVector[1]);
	double volatility = stod(inputDataVector[2]);
	double riskFreeRate = stod(inputDataVector[3]);
	Input_market_data inputMarket(initialPrice, volatility, riskFreeRate);

	// Input option data
	double strikePrice = stod(inputDataVector[4]);
	double timeToMaturity = stod(inputDataVector[5]);
	unsigned int numberOfIntervals = stoul(inputDataVector[6]);
	char optionType = inputDataVector[7].at(0);		// Get char in position 0, just as well since this is supposed to be a single char string
	Input_option_data inputOption(strikePrice, numberOfIntervals, timeToMaturity, optionType);

	// Input Monte Carlo data
	unsigned int totalNumberOfSimulations = stoul(inputDataVector[8]);
	Input_MC_data inputMC(totalNumberOfSimulations);
	unsigned int numberOfSimulationsPerThread = inputMC.GetNumberOfSimulationsPerThread(inputGPU);
	
	// Print input data (duh)
	PrintInputData(inputGPU, inputOption, inputMarket, inputMC);

	// Template path for invidual paths created in each thread
	Path pathTemplate(inputMarket, inputOption, initialPrice);
	
	// Mersenne random generator of unsigned ints, courtesy of C++11
	// For reproducibility, replace time(NULL) with a fixed seed
	mt19937 mersenneCoreGenerator(time(NULL));
	uniform_int_distribution<unsigned int> uniformDistribution(129, UINT_MAX);

	RNGCombinedGenerator *randomGenerators = new RNGCombinedGenerator[totalNumberOfThreads];
	for(unsigned int threadNumber=0; threadNumber<totalNumberOfThreads; ++threadNumber){
		randomGenerators[threadNumber].SetInternalState(uniformDistribution(mersenneCoreGenerator),
														uniformDistribution(mersenneCoreGenerator),
														uniformDistribution(mersenneCoreGenerator),
														uniformDistribution(mersenneCoreGenerator));
	}
	
	// Output MC per thread
	Output_MC_per_thread *threadOutputs = new Output_MC_per_thread[totalNumberOfThreads];
	
/*
	////////////// HOST-SIDE GENERATOR //////////////	
	cout << "Beginning device simulation through CPU..." << endl;
	// Simulating device function
	OptionPricingEvaluator_Host(inputGPU, inputOption, inputMarket, inputMC, pathTemplate, randomGenerators, threadOutputs);
	cout << endl;
	/////////////////////////////////////////////////
*/

///*
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
//*/
	
	
	// Compute results
	double monteCarloEstimatedPrice, monteCarloError;
	tie(monteCarloEstimatedPrice, monteCarloError) = EvaluateEstimatedPriceAndError(threadOutputs, totalNumberOfThreads);
	
	// Elapsed time is temporary, will be implemented later
	double elapsedTime = 0.;
	
	// Global output MC
	Output_MC_data outputMC(monteCarloEstimatedPrice, monteCarloError, elapsedTime);
	outputMC.CompleteEvaluationOfBlackScholes(inputOption, inputMarket);
	outputMC.PrintResults();
	cout << endl;
	
	// Trash bin section, where segfaults come to die
	delete[] randomGenerators;
	delete[] threadOutputs;
	
	return 0;
}
