#include <vector>		// vector
#include <tuple>		// tuple, tie, make_tuple
#include <string>		// string, stoul, stod, at
#include <fstream>		// ifstream

#include "Support_functions.cuh"

using namespace std;

// Main evaluators
__host__ __device__ void OptionPricingEvaluator_HostDev(Input_gpu_data inputGPU, Input_option_data option, Input_market_data market, Input_MC_data inputMC, Statistics* exactOutputs, Statistics* eulerOutputs, unsigned int threadNumber){
	
	unsigned int numberOfPathsPerThread = inputMC.GetNumberOfSimulationsPerThread(inputGPU);
	unsigned int numberOfIntervals = option.NumberOfIntervals;
	unsigned int totalNumberOfSimulations = inputMC.NumberOfMCSimulations;
	
	RNG *randomGenerator = new RNG_CombinedGenerator;
	randomGenerator->SetInternalState(11+threadNumber,1129+threadNumber,1130+threadNumber,1131+threadNumber);
	
	// Dummy variables to reduce memory accesses
	Path exactPath, eulerPath;
	double payoff;
	
	// Cycling through paths, overwriting the same dummy path with the same template path
	for(unsigned int pathNumber=0; pathNumber<numberOfPathsPerThread; ++pathNumber){
		// Check if we're not overflowing. Since we decide a priori the number of simulations, some threads will inevitably work less
		if(numberOfPathsPerThread * threadNumber + pathNumber < totalNumberOfSimulations){
			exactPath.ResetToInitialState(market, option);
			eulerPath.ResetToInitialState(market, option);
			
			// Cycling through steps in each path
			for(unsigned int stepNumber=0; stepNumber<numberOfIntervals; ++stepNumber){
				exactPath.ExactLogNormalStep(randomGenerator->GetGauss());
				eulerPath.EulerLogNormalStep(randomGenerator->GetGauss());
			}
			payoff = EvaluatePayoff(exactPath, option);
			exactOutputs[threadNumber].AddPayoff(ActualizePayoff(payoff, market.RiskFreeRate, option.TimeToMaturity));
			payoff = EvaluatePayoff(eulerPath, option);
			eulerOutputs[threadNumber].AddPayoff(ActualizePayoff(payoff, market.RiskFreeRate, option.TimeToMaturity));
		}
	}
}

__host__ void OptionPricingEvaluator_Host(Input_gpu_data inputGPU, Input_option_data option, Input_market_data market, Input_MC_data inputMC, Statistics* exactOutputs, Statistics* eulerOutputs){
	unsigned int totalNumberOfThreads = inputGPU.GetTotalNumberOfThreads();
	
	for(unsigned int threadNumber=0; threadNumber<totalNumberOfThreads; ++threadNumber)
		OptionPricingEvaluator_HostDev(inputGPU, option, market, inputMC, exactOutputs, eulerOutputs, threadNumber);
}

__global__ void OptionPricingEvaluator_Global(Input_gpu_data inputGPU, Input_option_data option, Input_market_data market, Input_MC_data inputMC, Statistics* exactOutputs, Statistics* eulerOutputs){
	unsigned int threadNumber = threadIdx.x + blockDim.x * blockIdx.x;
	OptionPricingEvaluator_HostDev(inputGPU, option, market, inputMC, exactOutputs, eulerOutputs, threadNumber);
}


// Support functions
__host__ __device__ double EvaluatePayoff(const Path& path, const Input_option_data& option){
	double spotPrice = path.GetSpotPrice();
	
	switch(option.OptionType){
		case 'f':
			return spotPrice;
		
		case 'c':
			{
				double strikePrice = option.StrikePrice;
				return fmax(spotPrice - strikePrice, 0.);
			}
		
		case 'p':
			{
				double strikePrice = option.StrikePrice;
				return fmax(strikePrice - spotPrice, 0.);
			}
		
		case 'e':
			{
				double K = option.K;
				double N = option.N;
				unsigned int performanceCorridorCounter = path.GetPerformanceCorridorBarrierCounter();
				unsigned int numberOfIntervals = option.NumberOfIntervals;
				return N * fmax((static_cast<double>(performanceCorridorCounter) / numberOfIntervals) - K, 0.);
			}
			
		default:
			return -10000.;
	}
}

__host__ __device__ double ActualizePayoff(double payoff, double riskFreeRate, double timeToMaturity){
	return (payoff * exp(- riskFreeRate*timeToMaturity));
}

__host__ void PrintInputData(const Input_gpu_data& inputGPU, const Input_option_data& option, const Input_market_data& market, const Input_MC_data& inputMC){
	cout << endl << "###### INPUT DATA ######" << endl << endl;
	cout << "## GPU AND MC INPUT DATA ##" << endl;
	cout << "Number of blocks: " << inputGPU.NumberOfBlocks << endl;
	cout << "Number of threads per block: " << inputGPU.GetNumberOfThreadsPerBlock() << endl;
	cout << "Total number of threads: " << inputGPU.GetTotalNumberOfThreads() << endl; 
	cout << "Number of simulations: " << inputMC.NumberOfMCSimulations << endl;
	cout << "Number of simulations per thread (round-up): " << inputMC.GetNumberOfSimulationsPerThread(inputGPU) << endl;
	
	cout << "## MARKET DATA ##" << endl;
	cout << "Initial underlying price [USD]: " << market.InitialPrice << endl;
	cout << "Market volatility: " << market.Volatility << endl;
	cout << "Risk free rate: " << market.RiskFreeRate << endl; 
	
	cout << "## OPTION DATA ##" << endl;
	cout << "Option type: " << option.OptionType << endl; 
	cout << "Time to option maturity [years]: " << option.TimeToMaturity << endl;
	cout << "Number of intervals for Euler/exact step-by-step computation: " << option.NumberOfIntervals << endl;
	cout << "Interval time [years]: " << option.GetDeltaTime() << endl;
	switch(option.OptionType){
		case 'p':
		case 'c':
			cout << "Option strike price [USD]: " << option.StrikePrice << endl;
			break;
		
		case 'e':
			cout << "B: " << option.B << endl;
			cout << "K [percentage]: " << option.K << endl;
			cout << "N [EUR]: " << option.N << endl;
		
		case 'f':
		default:
			break;
	}
	
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


// Output processing
__host__ void PrintOutputData(const Output_MC_data& outputMC){
	cout << "Monte Carlo estimated price via exact formula [EUR]: " << outputMC.EstimatedPriceMCExact << endl;
	cout << "Monte Carlo estimated error via exact formula [EUR]: " << outputMC.ErrorMCExact << endl;
	cout << "Monte Carlo estimated price via Euler formula [EUR]: " << outputMC.EstimatedPriceMCEuler << endl;
	cout << "Monte Carlo estimated error via Euler formula [EUR]: " << outputMC.ErrorMCEuler << endl;
	cout << "Computation time [ms]: " << outputMC.Tick << endl;
}
