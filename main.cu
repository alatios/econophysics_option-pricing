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
#include "libraries/OutputMCPerThread/Output_MC_per_thread.cuh"
#include "random_generator/rng.cuh"

using namespace std;

// Main evaluators
__host__ void OptionPricingEvaluator_Host(Input_gpu_data, Input_option_data, Input_market_data, Input_MC_data, Path, RNGCombinedGenerator*, Output_MC_per_thread*);
__host__ __device__ void OptionPricingEvaluator_HostDev(Input_gpu_data, Input_option_data, Input_market_data, Input_MC_data, Path, RNGCombinedGenerator*, Output_MC_per_thread*, unsigned int threadNumber);

// Support functions
__host__ __device__ double EvaluatePayoff(const Path&, const Input_option_data&);
__host__ __device__ double ActualizePayoff(double payoff, double riskFreeRate, double timeToMaturity);

int main(){
	
	// Input GPU data
	unsigned int numberOfBlocks = 10;
	Input_gpu_data inputGPU(numberOfBlocks);
	unsigned int numberOfThreadsPerBlock = inputGPU.GetNumberOfThreadsPerBlock();
	unsigned int totalNumberOfThreads = inputGPU.GetTotalNumberOfThreads();
	
	// Input market data
	double initialPrice = 100.;			// USD
	double volatility = 0.25;			// Percentage
	double riskFreeRate = 0.01;			// 50% per year (percentage per unit of time)
	Input_market_data inputMarket(initialPrice, volatility, riskFreeRate);

	// Input option data
	double strikePrice = 100.;				// $
	double timeToMaturity = 1.;				// years
	unsigned int numberOfIntervals = 365;	// No unit of measure
	char optionType = 'c';					// Call option
	Input_option_data inputOption(strikePrice, numberOfIntervals, timeToMaturity, optionType);


	// Input Monte Carlo data
	unsigned int totalNumberOfSimulations = 10000;
	Input_MC_data inputMC(totalNumberOfSimulations);
	unsigned int numberOfSimulationsPerThread = inputMC.GetNumberOfSimulationsPerThread(inputGPU);

	// Template path for invidual paths created in each thread
	Path pathTemplate(inputMarket, inputOption, initialPrice);
	
	// Mersenne random generator of unsigned ints, courtesy of C++11
	// For reproducibility, replace time(NULL) with a fixed seed
	mt19937 mersenneCoreGenerator(time(NULL));
	uniform_int_distribution<unsigned int> mersenneDistribution(129, UINT_MAX);

	RNGCombinedGenerator *randomGenerators = new RNGCombinedGenerator[totalNumberOfThreads];
	for(unsigned int threadNumber=0; threadNumber<totalNumberOfThreads; ++threadNumber){
		randomGenerators[threadNumber].SetSeedLCGS(mersenneDistribution(mersenneCoreGenerator));
		randomGenerators[threadNumber].SetSeedTaus1(mersenneDistribution(mersenneCoreGenerator));
		randomGenerators[threadNumber].SetSeedTaus2(mersenneDistribution(mersenneCoreGenerator));
		randomGenerators[threadNumber].SetSeedTaus3(mersenneDistribution(mersenneCoreGenerator));
	}
	
	// Output MC per thread
	Output_MC_per_thread *threadOutputs = new Output_MC_per_thread[totalNumberOfThreads];
	
	// Simulating device function
	OptionPricingEvaluator_Host(inputGPU, inputOption, inputMarket, inputMC, pathTemplate, randomGenerators, threadOutputs);
	
	// Sum all payoffs from threads, then average them
	double totalSumOfPayoffs = 0;
	double totalSumOfSquaredPayoffs = 0;
	
	for(unsigned int threadNumber=0; threadNumber<totalNumberOfThreads; ++threadNumber){
		totalSumOfPayoffs += threadOutputs[threadNumber].GetPayoffSum();
		totalSumOfSquaredPayoffs += threadOutputs[threadNumber].GetSquaredPayoffSum();
	}
		
	double monteCarloEstimatedPrice = totalSumOfPayoffs / totalNumberOfSimulations;
	double monteCarloError = sqrt(((totalSumOfSquaredPayoffs/totalNumberOfSimulations) - pow(monteCarloEstimatedPrice,2))/totalNumberOfSimulations);
	// Elapsed time is temporary, will be implemented later
	double elapsedTime = 0.;
	
	// Global output MC
	Output_MC_data outputMC(monteCarloEstimatedPrice, monteCarloError, elapsedTime);
	outputMC.CompleteEvaluationOfBlackScholes(inputOption, inputMarket);
	
	cout << "MC estimated price [USD] = " << outputMC.GetEstimatedPriceMC() << endl;
	cout << "MC error [USD] = " << outputMC.GetErrorMC() << endl;
	cout << "Elapsed time [ms] = " << outputMC.GetTick() << endl;
	cout << "Black-Scholes estimated price = " << outputMC.GetBlackScholesPrice() << endl;
	cout << "Black-Scholes discrepancy [MCSigmas] = " << outputMC.GetErrorBlackScholes() << endl;
	
	
	// Trash bin section, where segfaults come to die
	delete[] randomGenerators;
	delete[] threadOutputs;
	
	return 0;
}

//////////////////////////////////////////
//////////////////////////////////////////
//////   FUNCTIONS IMPLEMENTATION   //////
//////////////////////////////////////////
//////////////////////////////////////////

// Main evaluators
__host__ __device__ void OptionPricingEvaluator_HostDev(Input_gpu_data inputGPU, Input_option_data option, Input_market_data market, Input_MC_data inputMC, Path pathTemplate, RNGCombinedGenerator* randomGenerators, Output_MC_per_thread* threadOutputs, unsigned int threadNumber){
	
	unsigned int numberOfPathsPerThread = inputMC.GetNumberOfSimulationsPerThread(inputGPU);
	unsigned int numberOfIntervals = option.GetNumberOfIntervals();
	unsigned int totalNumberOfSimulations = inputMC.GetNumberOfMCSimulations();
	
	// Dummy variables to reduce memory accesses
	Path currentPath;
	double payoff, actualizedPayoff;
	
	// Cycling through paths, overwriting the same dummy path with the same template path
	for(unsigned int pathNumber=0; pathNumber<numberOfPathsPerThread; ++pathNumber){
		// Check if we're not overflowing. Since we decide a priori the number of simulations, some threads will inevitably work less
		if(numberOfPathsPerThread * threadNumber + pathNumber < totalNumberOfSimulations){
			currentPath.SetInternalState(pathTemplate);
			
			// Cycling on steps in each path
			for(unsigned int stepNumber=0; stepNumber<numberOfIntervals; ++stepNumber)
				currentPath.EuleroStep(randomGenerators[threadNumber].GetGauss());
			
			payoff = EvaluatePayoff(currentPath, option);
			actualizedPayoff = ActualizePayoff(payoff, market.GetRiskFreeRate(), option.GetTimeToMaturity());
			threadOutputs[threadNumber].AddToAll(actualizedPayoff);
		}
	}
}

__host__ void OptionPricingEvaluator_Host(Input_gpu_data inputGPU, Input_option_data option, Input_market_data market, Input_MC_data inputMC, Path pathTemplate, RNGCombinedGenerator* randomGenerators, Output_MC_per_thread* threadOutputs){
	unsigned int totalNumberOfThreads = inputGPU.GetTotalNumberOfThreads();
	
	for(unsigned int threadNumber=0; threadNumber<totalNumberOfThreads; ++threadNumber)
		OptionPricingEvaluator_HostDev(inputGPU, option, market, inputMC, pathTemplate, randomGenerators, threadOutputs, threadNumber);
}

// Support functions
__host__ __device__ double EvaluatePayoff(const Path& path, const Input_option_data& option){
	double spotPrice = path.GetSpotPrice();
	double strikePrice = option.GetStrikePrice();
	char optionType = option.GetOptionType();
	
	if(optionType == 'c')
		return fmax(spotPrice - strikePrice, 0.);
	else if(optionType == 'p')
		return fmax(strikePrice - spotPrice, 0.);
	else
		return -10000.;
}

__host__ __device__ double ActualizePayoff(double payoff, double riskFreeRate, double timeToMaturity){
	return (payoff * exp(- riskFreeRate*timeToMaturity));
}

