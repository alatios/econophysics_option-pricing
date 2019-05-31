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

#include "libraries/InputGPUData/Input_gpu_data.cuh"
#include "libraries/InputMarketData/Input_market_data.cuh"
#include "libraries/InputMCData/Input_MC_data.cuh"
#include "libraries/InputOptionData/Input_option_data.cuh"
#include "libraries/OutputMCData/Output_MC_data.cuh"
#include "libraries/Path/Path.cuh"
#include "libraries/PathPerThread/Path_per_thread.cuh"
#include "random_generator/rng.cuh"

using namespace std;

int main(){
	
	// Input GPU data
	unsigned int numberOfBlocks = 14;
	unsigned int numberOfThreadsPerBlock = 1024;
	unsigned int totalNumberOfThreads = numberOfBlocks * numberOfThreadsPerBlock;
	
	Input_gpu_data inputGPU(numberOfBlocks, numberOfThreadsPerBlock);
	
	// Input market data
	float zeroPrice = 100.;		// $
	float volatility = 0.25;	// No unit of measure
	float riskFreeRate = 0.1;	// 10%
	Input_market_data inputMarket(zeroPrice, volatility, riskFreeRate);
	
	// Input option data
	float strikePrice = 110.;				// $
	float timeToMaturity = 365.				// days
	unsigned int numberOfIntervals = 365;	// No unit of measure
	char optionType = 'c';					// Call option
	Input_option_data inputOption(strikePrice, numberOfIntervals, timeToMaturity);
	
	// Input Monte Carlo data
	unsigned int numberOfSimulationsPerThread = 5;
	Input_MC_data inputMC(numberOfSimulationsPerThread, inputGPU);
	
	// Output Monte Carlo data (default, will be set in global function)
	Output_MC_data outputMC;
	
	// Path per thread
	Path_per_thread *pathsPerThread = new Path_per_thread[totalNumberOfThreads];
	for(unsigned int i=0; i<totalNumberOfThreads; ++i){
//		pathsPerThread[i].SetNumberOfPathsPerThread(numberOfSimulationsPerThread);

		

		pathsPerThread[i].SetInputMarketData(inputMarket);
		pathsPerThread[i].SetInputOptionData(inputOption);
		pathsPerThread[i].SetSpotSprice(zeroPrice);
	}
	
	// Mersenne random generator of unsigned ints, courtesy of C++11
	// For reproducibility, replace time(NULL) with a fixed seed
	mt19937 mersenneCoreGenerator(time(NULL));
	uniform_int_distribution<unsigned int> mersenneDistribution(0, UINT_MAX);

	RNGCombinedGenerator *randomGenerators = new RNGCombinedGenerator[totalNumberOfThreads];
	for(unsigned int i=0; i<totalNumberOfThreads; ++i){
		randomGenerators[i].SetSeedLCGS(mersenneDistribution(mersenneCoreGenerator));
		randomGenerators[i].SetSeedTaus1(mersenneDistribution(mersenneCoreGenerator));
		randomGenerators[i].SetSeedTaus2(mersenneDistribution(mersenneCoreGenerator));
		randomGenerators[i].SetSeedTaus3(mersenneDistribution(mersenneCoreGenerator));
	}
	
	
	
	return 0;
}
