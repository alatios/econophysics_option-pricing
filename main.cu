#include <iostream>
#include <cstdlib>
#include <ctime>		// time(NULL) for seed
#include <random>		// C++11 Mersenne twister
#include <climits>		// UINT_MAX
#include <cmath>		// log, cos, sin, ceil, M_PI
#include <algorithm>	// min
#include <fstream>
#include <cstdio>

#include "libraries/InputGPUData/Input_gpu_data.cuh"
#include "libraries/InputMarketData/Input_market_data.cuh"
#include "libraries/InputMCData/Input_MC_data.cuh"
#include "libraries/InputOptionData/Input_option_data.cuh"
#include "libraries/OutputMCData/Output_MC_data.cuh"
#include "libraries/Path/Path.cuh"
#include "libraries/PathPerThread/Path_per_thread.cuh"
#include "random_generator/rng.cuh"

using namespace std;

// ARGS: thread no., InputMC object, Path and RNG arrays (one element per thread)
//__host__ __device__ void OptionPricingEvaluator_HostDev(unsigned int, Input_MC_data, Path**, RNGCombinedGenerator*);
__host__ void OptionPricingEvaluator_HostDev(unsigned int, Input_MC_data, Path**, RNGCombinedGenerator*);

// ARGS: number of blocks, number of threads per block, InputMC object, Path and RNG arrays (one element per thread)
__host__ void OptionPricingEvaluator_Host(unsigned int, unsigned int, Input_MC_data, Path**, RNGCombinedGenerator*);


int main(){
	
	// Input GPU data
	unsigned int numberOfBlocks = 2;
	unsigned int numberOfThreadsPerBlock = 512;
	unsigned int numberOfSimulationsPerThread = 50;
	unsigned int totalNumberOfThreads = numberOfBlocks * numberOfThreadsPerBlock;
	
	Input_gpu_data inputGPU(numberOfBlocks, numberOfThreadsPerBlock);
	
	// Input market data
	float zeroPrice = 100.;				// USD
	float volatility = 0.25;			// Percentage
	float riskFreeRate = 0.5;			// 50% per year (percentage per unit of time)
	Input_market_data inputMarket(zeroPrice, volatility, riskFreeRate);
	
	// Input option data
	float strikePrice = 110.;				// $
	float timeToMaturity = 1.;				// years
	unsigned int numberOfIntervals = 365;	// No unit of measure
	char optionType = 'c';					// Call option
	Input_option_data inputOption(strikePrice, numberOfIntervals, timeToMaturity, optionType);
	
	// Input Monte Carlo data
	Input_MC_data inputMC(numberOfSimulationsPerThread, inputGPU);
	
	// Output Monte Carlo data (default, will be set in global function)
	Output_MC_data outputMC;
	
	// Path per thread
	Path pathTemplate(inputMarket, inputOption, zeroPrice);

/*	
	Path_per_thread **pathsPerThread = new Path_per_thread*[totalNumberOfThreads];
	for(unsigned int threadNumber=0; threadNumber<totalNumberOfThreads; ++threadNumber){
		pathsPerThread[threadNumber] = new Path_per_thread(numberOfSimulationsPerThread);
		
		for(unsigned int simulationNumber=0; simulationNumber<numberOfSimulationsPerThread; ++simulationNumber)
			pathsPerThread[threadNumber]->SetPathComponent(simulationNumber, pathTemplate);
	}
*/

	Path **paths = new Path*[totalNumberOfThreads];
	for(unsigned int threadNumber=0; threadNumber<totalNumberOfThreads; ++threadNumber)
		paths[threadNumber] = new Path(inputMarket, inputOption, zeroPrice);

	
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
	cout << "thread\t path\t interval\t spotprice" << endl;
	OptionPricingEvaluator_Host(numberOfBlocks, numberOfThreadsPerBlock, inputMC, paths, randomGenerators);
	
	
	// Trash bin section, where segfaults come to die
	for(unsigned int threadNumber=0; threadNumber<totalNumberOfThreads; ++threadNumber)
		delete paths[threadNumber];
	
	delete[] paths;
	delete[] randomGenerators;
	
	return 0;
}

///////////////////////////
///////////////////////////
//////   FUNCTIONS   //////
///////////////////////////
///////////////////////////

//__host__ __device__ void OptionPricingEvaluator_HostDev(unsigned int threadNumber, Path_per_thread** pathsPerThread, RNGCombinedGenerator* randomGenerators){
__host__ void OptionPricingEvaluator_HostDev(unsigned int threadNumber, Input_MC_data inputMC, Path** paths, RNGCombinedGenerator* randomGenerators){
	unsigned int numberOfPathsPerThread = inputMC.GetNumberOfMCSimulationsPerThread();
	unsigned int numberOfIntervalsPerPath = (paths[threadNumber]->GetInputOptionData()).GetNumberOfIntervals();
	
	// Dummy variable to reduce memory access
	Path currentPath;
	
	// Cycling on paths
	for(unsigned int pathNumber=0; pathNumber<3; ++pathNumber){
//	for(unsigned int pathNumber=0; pathNumber<numberOfPathsPerThread; ++pathNumber){
		currentPath.SetInputMarketData(paths[threadNumber]->GetInputMarketData());
		currentPath.SetInputOptionData(paths[threadNumber]->GetInputOptionData());
		currentPath.SetSpotPrice(paths[threadNumber]->GetSpotPrice());
		
		cout << threadNumber << "\t" << pathNumber << "\t" << 0 << "\t" << currentPath.GetSpotPrice() << endl;
	
		// Cycling on steps in each path
		for(unsigned int stepNumber=0; stepNumber<numberOfIntervalsPerPath; ++stepNumber){
			currentPath.SetGaussianRandomVariable(randomGenerators[threadNumber].GetGauss());
			currentPath.EuleroStep();
			cout << threadNumber << "\t" << pathNumber << "\t" << stepNumber+1 << "\t" << currentPath.GetSpotPrice() << endl;
		}
	}
}

__host__ void OptionPricingEvaluator_Host(unsigned int numberOfBlocks, unsigned int numberOfThreadsPerBlock, Input_MC_data inputMC, Path** paths, RNGCombinedGenerator* randomGenerators){
	unsigned int totalNumberOfThreads = numberOfBlocks * numberOfThreadsPerBlock;
	
	for(unsigned int threadNumber=0; threadNumber<totalNumberOfThreads; ++threadNumber)
		OptionPricingEvaluator_HostDev(threadNumber, inputMC, paths, randomGenerators);
}
