#ifndef __Output_MC_data_h__
#define __Output_MC_data_h__

#include <iostream>
#include <cmath>
#include "../InputMarketData/Input_market_data.cuh"
#include "../InputOptionData/Input_option_data.cuh"

using namespace std;

class Output_MC_data{

private:

	Input_market_data _MarketData;
	Input_option_data _OptionData;

	float _EstimatedPriceMC;
	float _ErrorMC;
	float _ErrorBlackScholes;				//error with respect to the exact result
	float _Tick;							//calculation time [ms]
	float _BlackScholesPrice;
	
	__device__ __host__ void BlackScholesCallOption();
	__device__ __host__ void BlackScholesPutOption();

public:

	__device__ __host__ Output_MC_data();
	__device__ __host__ Output_MC_data(const Input_market_data&, const Input_option_data&, float, float, float);
	__device__ __host__ ~Output_MC_data() = default;

	__device__ __host__ void SetEstimatedPriceMC(float);
	__device__ __host__ float GetEstimatedPriceMC() const;
	__device__ __host__ void SetErrorMC(float);
	__device__ __host__ float GetErrorMC() const;
	__device__ __host__ void SetErrorBlackScholes();
	__device__ __host__ float GetErrorBlackScholes();
	__device__ __host__ void SetTick(float);
	__device__ __host__ float GetTick() const;
	__device__ __host__ void SetBlackScholesPrice();
	__device__ __host__ float GetBlackScholesPrice();
	
	__device__ __host__ Input_market_data GetInputMarketData() const;
	__device__ __host__ void SetInputMarketData(const Input_market_data&);
	__device__ __host__ Input_option_data GetInputOptionData() const;
	__device__ __host__ void SetInputOptionData(const Input_option_data&);
};
#endif
