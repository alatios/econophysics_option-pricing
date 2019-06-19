#include <vector>		// vector
#include <tuple>		// tuple, tie, make_tuple
#include <string>		// string, stoul, stod, at
#include <fstream>		// ifstream

#include "Support_functions.cuh"

using namespace std;

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

__global__ void OptionPricingEvaluator_Global(Input_gpu_data inputGPU, Input_option_data option, Input_market_data market, Input_MC_data inputMC, Path pathTemplate, RNGCombinedGenerator* randomGenerators, Output_MC_per_thread* threadOutputs){
	unsigned int threadNumber = threadIdx.x + blockDim.x * blockIdx.x;
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
	
	// Sometimes, infinities occur. No idea why.
	unsigned int infinityCounter = 0;
	
	for(unsigned int threadNumber=0; threadNumber<totalNumberOfThreads; ++threadNumber){
		
		// Check if both values are finite. std:: is needed on our machine because of C++11 compatibility issues.
		if(std::isinf(threadOutputs[threadNumber].GetPayoffSum()) || std::isinf(threadOutputs[threadNumber].GetSquaredPayoffSum()))
			continue;

		totalSumOfPayoffs += threadOutputs[threadNumber].GetPayoffSum();
		totalPayoffCounter += threadOutputs[threadNumber].GetPayoffCounter();
		totalSumOfSquaredPayoffs += threadOutputs[threadNumber].GetSquaredPayoffSum();
		totalSquaredPayoffCounter += threadOutputs[threadNumber].GetSquaredPayoffCounter();
	}
	
	if(!infinityCounter)
		cout << infinityCounter << " thread(s) with infinite payoffs or squared payoffs were encountered (and skipped) during this run." << endl << endl;
	
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
