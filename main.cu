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

// ARGS: thread no., PPT and RNG arrays (one element per thread)
//__host__ __device__ void OptionPricingEvaluator_HostDev(unsigned int, Path_per_thread**, RNGCombinedGenerator*);
__host__ void OptionPricingEvaluator_HostDev(unsigned int, Path_per_thread**, RNGCombinedGenerator*);

// ARGS: number of blocks, number of threads per block, PPT and RNG arrays (one element per thread)
__host__ void OptionPricingEvaluator_Host(unsigned int, unsigned int, Path_per_thread**, RNGCombinedGenerator*);


int main(){
	
	// Input GPU data
	unsigned int numberOfBlocks = 2;
	unsigned int numberOfThreadsPerBlock = 512;
	unsigned int numberOfSimulationsPerThread = 50;
	unsigned int totalNumberOfThreads = numberOfBlocks * numberOfThreadsPerBlock;
	
	Input_gpu_data inputGPU(numberOfBlocks, numberOfThreadsPerBlock);
	
	// Input market data
	float zeroPrice = 100.;		// $
	float volatility = 0.25;	// No unit of measure
	float riskFreeRate = 0.1;	// 10%
	Input_market_data inputMarket(zeroPrice, volatility, riskFreeRate);
	
	// Input option data
	float strikePrice = 110.;				// $
	float timeToMaturity = 365.;			// days
	unsigned int numberOfIntervals = 365;	// No unit of measure
	char optionType = 'c';					// Call option
	Input_option_data inputOption(strikePrice, numberOfIntervals, timeToMaturity, optionType);
	
	// Input Monte Carlo data
	Input_MC_data inputMC(numberOfSimulationsPerThread, inputGPU);
	
	// Output Monte Carlo data (default, will be set in global function)
	Output_MC_data outputMC;
	
	// Path per thread
	Path pathTemplate(inputMarket, inputOption, zeroPrice);
	
	Path_per_thread **pathsPerThread = new Path_per_thread*[totalNumberOfThreads];
	for(unsigned int threadNumber=0; threadNumber<totalNumberOfThreads; ++threadNumber){
		pathsPerThread[threadNumber] = new Path_per_thread(numberOfSimulationsPerThread);
		
		for(unsigned int simulationNumber=0; simulationNumber<numberOfSimulationsPerThread; ++simulationNumber)
			pathsPerThread[threadNumber]->SetPathComponent(simulationNumber, pathTemplate);
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
	
	// Simulating device function
	cout << "thread\t path\t interval" << endl;
	OptionPricingEvaluator_Host(numberOfBlocks, numberOfThreadsPerBlock, pathsPerThread, randomGenerators);
	
	
	// Trash bin section, where segfaults come to die
	for(unsigned int threadNumber=0; threadNumber<totalNumberOfThreads; ++threadNumber)
		delete pathsPerThread[threadNumber];
	
	delete[] pathsPerThread;
	delete[] randomGenerators;
	
	return 0;
}

///////////////////////////
///////////////////////////
//////   FUNCTIONS   //////
///////////////////////////
///////////////////////////

//__host__ __device__ void OptionPricingEvaluator_HostDev(unsigned int threadNumber, Path_per_thread** pathsPerThread, RNGCombinedGenerator* randomGenerators){
__host__ void OptionPricingEvaluator_HostDev(unsigned int threadNumber, Path_per_thread** pathsPerThread, RNGCombinedGenerator* randomGenerators){
	unsigned int numberOfPathsPerThread = pathsPerThread[threadNumber]->GetNumberOfPathsPerThread();
	unsigned int numberOfIntervalsPerPath = 365;
	
	for(unsigned int pathNumber=0; pathNumber<numberOfPathsPerThread; ++pathNumber){
		for(unsigned int stepNumber=0; stepNumber<numberOfIntervalsPerPath; ++stepNumber){
			cout << threadNumber << "\t" << pathNumber << "\t" << stepNumber << endl;
		}
	}
}

__host__ void OptionPricingEvaluator_Host(unsigned int numberOfBlocks, unsigned int numberOfThreadsPerBlock, Path_per_thread** pathsPerThread, RNGCombinedGenerator* randomGenerators){
	unsigned int totalNumberOfThreads = numberOfBlocks * numberOfThreadsPerBlock;
	
	for(unsigned int threadNumber=0; threadNumber<totalNumberOfThreads; ++threadNumber)
		OptionPricingEvaluator_HostDev(threadNumber, pathsPerThread, randomGenerators);
}
