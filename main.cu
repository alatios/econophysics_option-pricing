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
#include "libraries/OutputMCPerThread/Output_MC_per_thread.cuh"
#include "random_generator/rng.cuh"

using namespace std;

// Main evaluators
__host__ void OptionPricingEvaluator_Host(Input_gpu_data, Input_option_data, Input_market_data, Input_MC_data, Path, RNGCombinedGenerator*, Output_MC_per_thread*);
__host__ __device__ void OptionPricingEvaluator_HostDev(Input_gpu_data, Input_option_data, Input_market_data, Input_MC_data, Path, RNGCombinedGenerator*, Output_MC_per_thread*, unsigned int threadNumber);

// Support functions
__host__ __device__ double EvaluatePayoff(const Path&, const Input_option_data&);
__host__ __device__ double ActualizePayoff(double payoff, double riskFreeRate, double timeToMaturity);
__host__ tuple<double, double> EvaluateEstimatedPriceAndError(Output_MC_per_thread*, unsigned int totalNumberOfThreads);
__host__ void PrintInputData(const Input_gpu_data&, const Input_option_data&, const Input_market_data&, const Input_MC_data&);
__host__ void ReadInputData(vector<string>&, string sourceFile);

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
	
	
	cout << "Beginning device simulation through CPU..." << endl;
	// Simulating device function
	OptionPricingEvaluator_Host(inputGPU, inputOption, inputMarket, inputMC, pathTemplate, randomGenerators, threadOutputs);
	cout << endl;
	
	
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

__host__ tuple<double, double> EvaluateEstimatedPriceAndError(Output_MC_per_thread* threadOutputs, unsigned int totalNumberOfThreads){
	double totalSumOfPayoffs = 0;
	double totalSumOfSquaredPayoffs = 0;
	unsigned int totalPayoffCounter = 0;
	unsigned int totalSquaredPayoffCounter = 0;
	
	for(unsigned int threadNumber=0; threadNumber<totalNumberOfThreads; ++threadNumber){
		totalSumOfPayoffs += threadOutputs[threadNumber].GetPayoffSum();
		totalPayoffCounter += threadOutputs[threadNumber].GetPayoffCounter();
		totalSumOfSquaredPayoffs += threadOutputs[threadNumber].GetSquaredPayoffSum();
		totalSquaredPayoffCounter += threadOutputs[threadNumber].GetSquaredPayoffCounter();
	}
	
	if(totalPayoffCounter != totalSquaredPayoffCounter)
		cerr << "WARNING: Count of payoffs and squared payoffs in EvaluateEstimatedPriceAndError() are not equal." << endl;
		
	double monteCarloEstimatedPrice = totalSumOfPayoffs / totalPayoffCounter;
	double monteCarloError = sqrt(((totalSumOfSquaredPayoffs/totalSquaredPayoffCounter) - pow(monteCarloEstimatedPrice,2))/totalSquaredPayoffCounter);	
	return make_tuple(monteCarloEstimatedPrice, monteCarloError);
}

__host__ void PrintInputData(const Input_gpu_data& inputGPU, const Input_option_data& option, const Input_market_data& market, const Input_MC_data& inputMC){
	cout << endl << "###### INPUT DATA ######" << endl << endl;
	cout << "## GPU AND MC INPUT DATA ##" << endl;
	inputGPU.PrintGPUInput();
	inputMC.PrintMCInput(inputGPU);

	cout << "## MARKET DATA ##" << endl;
	market.PrintMarketInput();
	
	cout << "## OPTION DATA ##" << endl;
	option.PrintOptionInput();
	
	cout << endl;
}

__host__ void ReadInputData(vector<string>& inputDataVector, string sourceFile){
	ifstream inputFileStream(sourceFile.c_str());
	string line;
	if(inputFileStream.is_open()){
		while(getline(inputFileStream, line))
			if(line[0] != '#')
				inputDataVector.push_back(line);
	}else
		cout << "ERROR: Unable to open file " << sourceFile << "." << endl;
}
