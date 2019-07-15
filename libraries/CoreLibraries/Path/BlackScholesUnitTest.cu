#include <iostream>
#include <iomanip>

#include "../../InputStructures/InputMarketData/Input_market_data.cuh"
#include "../../InputStructures/InputOptionData/Input_option_data.cuh"
#include "../../InputStructures/InputGPUData/Input_gpu_data.cuh"
#include "../../InputStructures/InputMCData/Input_MC_data.cuh"
#include "../../CoreLibraries/DataStreamManager/Data_stream_manager.cuh"
#include "../../CoreLibraries/SupportFunctions/Support_functions.cuh"
#include "Path.cuh"

using namespace std;

// Black & Scholes formula
__host__ double GetBlackAndScholesCallPrice(const Input_market_data& inputMarket, const Input_option_data& inputOption){
	double d1 = 1./(inputMarket.Volatility * sqrt(inputOption.TimeToMaturity)) 
	* (log(inputMarket.InitialPrice / inputOption.StrikePrice)
	+ (inputMarket.RiskFreeRate + pow(inputMarket.Volatility,2)/2) * inputOption.TimeToMaturity);

	double d2 = d1 -  inputMarket.Volatility * sqrt(inputOption.TimeToMaturity);

	double callPrice = inputMarket.InitialPrice * (0.5 * (1. + erf(d1/sqrt(2.)))) - inputOption.StrikePrice 
	* exp(- inputMarket.RiskFreeRate * inputOption.TimeToMaturity)
	* (0.5 * (1. + erf(d2/sqrt(2.))));

	return callPrice;	 
}

__host__ double GetBlackAndScholesPutPrice(const Input_market_data& inputMarket, const Input_option_data& inputOption){
	double d1 = 1./(inputMarket.Volatility * sqrt(inputOption.TimeToMaturity)) 
	* (log(inputMarket.InitialPrice / inputOption.StrikePrice)
	+ (inputMarket.RiskFreeRate + pow(inputMarket.Volatility,2)/2) * inputOption.TimeToMaturity);

	double d2 = d1 -  inputMarket.Volatility * sqrt(inputOption.TimeToMaturity);

	double putPrice = inputMarket.InitialPrice * ((0.5 * (1. + erf(d1/sqrt(2.)))) - 1) - inputOption.StrikePrice
	* exp(- inputMarket.RiskFreeRate * inputOption.TimeToMaturity)
	* ((0.5 * (1. + erf(d2/sqrt(2.)))) - 1);

	return putPrice;

}

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

	if(inputOption.OptionType == char('c')) cout << "Black and Scholes Call price: " << setprecision(20) << GetBlackAndScholesCallPrice(inputMarket, inputOption) << endl << endl;
	if(inputOption.OptionType == char('p')) cout << "Black and Scholes Put price: " << setprecision(20) << GetBlackAndScholesPutPrice(inputMarket, inputOption) << endl << endl;
	
    return 0;
}