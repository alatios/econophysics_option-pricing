#include <fstream>	// ifstream
#include <string>	// string, stoul, stod, at
#include <vector>	// vector

#include "Data_stream_manager.cuh"

using namespace std;

// Constructors
__host__ Data_stream_manager::Data_stream_manager(){
	_InputFile = "blank";
}
__host__ Data_stream_manager::Data_stream_manager(string inputFile){
	_InputFile = inputFile;
}

// Set input file
__host__ void Data_stream_manager::SetInputFile(string inputFile){
	_InputFile = inputFile;	
}

// Input processing
__host__ void Data_stream_manager::ReadInputData(Input_gpu_data& inputGPU, Input_option_data& inputOption, Input_market_data& inputMarket, Input_MC_data& inputMC) const{
	vector<string> inputDataVector;
	ifstream inputFileStream(_InputFile.c_str());
	string line;
	if(inputFileStream.is_open()){
		while(getline(inputFileStream, line))
			if(line[0] != '#')
				inputDataVector.push_back(line);
	}else
		cout << "ERROR: Unable to open file " << _InputFile << "." << endl;
		
	// Input GPU data
	inputGPU.NumberOfBlocks = stoul(inputDataVector[0]);

	// Input market data
	inputMarket.InitialPrice = stod(inputDataVector[1]);
	inputMarket.Volatility = stod(inputDataVector[2]);
	inputMarket.RiskFreeRate = stod(inputDataVector[3]);

	// Input option data
	inputOption.TimeToMaturity = stod(inputDataVector[4]);
	inputOption.NumberOfIntervals = stoul(inputDataVector[5]);
	inputOption.OptionType = inputDataVector[6].at(0);
	inputOption.StrikePrice = stod(inputDataVector[7]);
	inputOption.B = stod(inputDataVector[8]);
	inputOption.K = stod(inputDataVector[9]);
	inputOption.N = stod(inputDataVector[10]);

	// Input Monte Carlo data
	inputMC.NumberOfMCSimulations = stoul(inputDataVector[11]);
	inputMC.CpuOrGpu = inputDataVector[12].at(0);
}

__host__ void Data_stream_manager::PrintInputData(const Input_gpu_data& inputGPU, const Input_option_data& inputOption, const Input_market_data& inputMarket, const Input_MC_data& inputMC) const{
	cout << endl << "###### INPUT DATA ######" << endl << endl;
	cout << "## GPU AND MC INPUT DATA ##" << endl;
	cout << "Number of blocks: " << inputGPU.NumberOfBlocks << endl;
	cout << "Number of threads per block: " << inputGPU.GetNumberOfThreadsPerBlock() << endl;
	cout << "Total number of threads: " << inputGPU.GetTotalNumberOfThreads() << endl; 
	cout << "Number of simulations: " << inputMC.NumberOfMCSimulations << endl;
	cout << "Number of simulations per thread (round-up): " << inputMC.GetNumberOfSimulationsPerThread(inputGPU) << endl;
	cout << "CPU v. GPU parameter: " << inputMC.CpuOrGpu << endl;
	
	cout << "## MARKET DATA ##" << endl;
	cout << "Initial underlying price [USD]: " << inputMarket.InitialPrice << endl;
	cout << "Market volatility: " << inputMarket.Volatility << endl;
	cout << "Risk free rate: " << inputMarket.RiskFreeRate << endl; 
	
	cout << "## OPTION DATA ##" << endl;
	cout << "Option type: " << inputOption.OptionType << endl; 
	cout << "Time to option maturity [years]: " << inputOption.TimeToMaturity << endl;
	cout << "Number of intervals for Euler/exact step-by-step computation: " << inputOption.NumberOfIntervals << endl;
	cout << "Interval time [years]: " << inputOption.GetDeltaTime() << endl;
	switch(inputOption.OptionType){
		case 'p':
		case 'c':
			cout << "Option strike price [USD]: " << inputOption.StrikePrice << endl;
			break;
		
		case 'e':
			cout << "B: " << inputOption.B << endl;
			cout << "K [percentage]: " << inputOption.K << endl;
			cout << "N [EUR]: " << inputOption.N << endl;
		
		case 'f':
		default:
			break;
	}
	
	cout << endl;	
}

// Output processing
__host__ void Data_stream_manager::StoreOutputData(Output_MC_data& outputMC, const Statistics exactResults, const Statistics eulerResults, double elapsedTime, char hostOrDevice) const{
	outputMC.EstimatedPriceMCExact = exactResults.GetPayoffAverage();
	outputMC.ErrorMCExact = exactResults.GetPayoffError();
	outputMC.EstimatedPriceMCEuler = eulerResults.GetPayoffAverage();
	outputMC.ErrorMCEuler = eulerResults.GetPayoffError();
	outputMC.Tick = elapsedTime;
	outputMC.HostOrDevice = hostOrDevice;
}

__host__ void Data_stream_manager::PrintOutputData(const Output_MC_data& outputMC) const{
	cout << endl << "## ";
	if(outputMC.HostOrDevice == 'h')
		cout << "HOST";
	else if(outputMC.HostOrDevice == 'd')
		cout << "DEVICE";
	else
		cout << "MISSINGNO.";
	
	cout << " OUTPUT MONTE CARLO DATA ##" << endl;
	cout << "Monte Carlo estimated price via exact formula [EUR]: " << outputMC.EstimatedPriceMCExact << endl;
	cout << "Monte Carlo estimated error via exact formula [EUR]: " << outputMC.ErrorMCExact << endl;
	cout << "Monte Carlo relative error via exact formula [EUR]: " << outputMC.GetRelativeErrorExact() << endl;
	cout << "Monte Carlo estimated price via Euler formula [EUR]: " << outputMC.EstimatedPriceMCEuler << endl;
	cout << "Monte Carlo estimated error via Euler formula [EUR]: " << outputMC.ErrorMCEuler << endl;
	cout << "Monte Carlo relative error via Euler formula [EUR]: " << outputMC.GetRelativeErrorEuler() << endl;
	cout << "Computation time [ms]: " << outputMC.Tick << endl;
	
	cout << endl;	
}
