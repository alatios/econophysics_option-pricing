#include <iostream>

#include "Data_stream_manager.cuh"
#include "../../InputStructures/InputGPUData/Input_gpu_data.cuh"
#include "../../InputStructures/InputMarketData/Input_market_data.cuh"
#include "../../InputStructures/InputMCData/Input_MC_data.cuh"
#include "../../InputStructures/InputOptionData/Input_option_data.cuh"
#include "../Statistics/Statistics.cuh"
#include "../../OutputStructures/OutputMCData/Output_MC_data.cuh"

using namespace std;

int main(){

	Data_stream_manager data_stream_manager("input.dat");
	Input_gpu_data inputGPU;
	Input_market_data inputMarket;
	Input_option_data inputOption;
	Input_MC_data inputMC;
	data_stream_manager.ReadInputData(inputGPU, inputOption, inputMarket, inputMC);

	bool test;

	cout << "\n-------------Data_stream_manager_test-------------\n";
	cout << "\nMethods testing\n";

	// Input GPU data
	test = (inputGPU.GetNumberOfThreadsPerBlock() == static_cast<unsigned int>(512));
	cout << test << "\t";
	test = (inputGPU.GetTotalNumberOfThreads() == static_cast<unsigned int>(5120));
	cout << test << "\n";
    
	// Input market data
	test = (inputMarket.InitialPrice == static_cast<double>(100.));
	cout << test << "\t";
	test = (inputMarket.Volatility == static_cast<double>(0.25));
	cout << test << "\t";
	test = (inputMarket.RiskFreeRate == static_cast<double>(0.01));
 	cout << test << "\n";

    	// Input option data
    	test = (inputOption.OptionType == char('e'));
	cout << test << "\t";
	test = (inputOption.NumberOfIntervals == static_cast<unsigned int>(365));
	cout << test << "\t";
	test = (inputOption.TimeToMaturity == static_cast<double>(1.));
	cout << test << "\t";
	test = (inputOption.StrikePrice == static_cast<double>(100.));
	cout << test << "\t";
    	test = (inputOption.B == static_cast<double>(1.));
	cout << test << "\t";
	test = (inputOption.N == static_cast<double>(1.));
	cout << test << "\t";
    	test = (inputOption.K == static_cast<double>(0.3));
    	cout << test << "\n";

    	// Input Monte Carlo data
    	test = (inputMC.GetNumberOfSimulationsPerThread(inputGPU) == static_cast<unsigned int>(977));
	cout << test << "\n";

    	return 0;
}
