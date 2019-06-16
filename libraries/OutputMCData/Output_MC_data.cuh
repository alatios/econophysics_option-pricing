#ifndef __Output_MC_data_h__
#define __Output_MC_data_h__

#include <iostream>
#include <cmath>
#include "../InputMarketData/Input_market_data.cuh"
#include "../InputOptionData/Input_option_data.cuh"

using namespace std;

class Output_MC_data{

private:

	double _EstimatedPriceMC;
	double _ErrorMC;
	double _BlackScholesPrice;
	double _ErrorBlackScholes;				//error with respect to the exact result
	double _Tick;							//calculation time [ms]
	
	__device__ __host__ void BlackScholesCallOption(const Input_option_data&, const Input_market_data&);
	__device__ __host__ void BlackScholesPutOption(const Input_option_data&, const Input_market_data&);

	__device__ __host__ void SetErrorBlackScholes(const Input_option_data&, const Input_market_data&);

public:

	__device__ __host__ Output_MC_data();
	__device__ __host__ Output_MC_data(const Input_option_data&, const Input_market_data&, double, double, double);
	__device__ __host__ ~Output_MC_data() = default;

	__device__ __host__ void SetEstimatedPriceMC(double);
	__device__ __host__ double GetEstimatedPriceMC() const;
	__device__ __host__ void SetErrorMC(double);
	__device__ __host__ double GetErrorMC() const;
	__device__ __host__ void EvaluateErrorBlackScholes(const Input_option_data&, const Input_market_data&);
	__device__ __host__ double GetErrorBlackScholes();
	__device__ __host__ void SetTick(double);
	__device__ __host__ double GetTick() const;
	__device__ __host__ void SetBlackScholesPrice(const Input_option_data&, const Input_market_data&);
	__device__ __host__ double GetBlackScholesPrice(const Input_option_data&, const Input_market_data&);
};
#endif
