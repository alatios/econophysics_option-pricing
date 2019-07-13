#include <iostream>
#include <iomanip>

#include "../../InputStructures/InputMarketData/Input_market_data.cuh"
#include "../../InputStructures/InputOptionData/Input_option_data.cuh"
#include "../../InputStructures/InputGPUData/Input_gpu_data.cuh"
#include "../../InputStructures/InputMCData/Input_MC_data.cuh"
#include "../../CoreLibraries/DataStreamManager/Data_stream_manager.cuh"
#include "Path.cuh"
#include "../../CoreLibraries/SupportFunctions/Support_functions.cuh"

using namespace std;

int main(){
    
    cout << "\n-------------Black and Scholes test-------------\n";

	// Read & print input data from file
	Data_stream_manager streamManager("input.dat");
	
	Input_gpu_data inputGPU;
	Input_option_data inputOption;
	Input_market_data inputMarket;
	Input_MC_data inputMC;
	streamManager.ReadInputData(inputGPU, inputOption, inputMarket, inputMC);
    streamManager.PrintInputData(inputGPU, inputOption, inputMarket, inputMC);
    
    Path path(inputMarket, inputOption);

    cout << "Black and Scholes price: " << setprecision(20) << path.GetBlackAndScholesPrice() << endl << endl;

    return 0;
}