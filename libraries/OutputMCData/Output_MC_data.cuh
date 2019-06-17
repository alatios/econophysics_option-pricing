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

		__device__ __host__ void SetBlackScholesPrice(double BlackScholesPrice);
		__device__ __host__ void SetErrorBlackScholes(double errorBlackScholes);

		__device__ __host__ void EvaluateBlackScholesPrice(const Input_option_data&, const Input_market_data&);
		__device__ __host__ void EvaluateErrorBlackScholes();

	public:

		__device__ __host__ Output_MC_data();
		__device__ __host__ Output_MC_data(double EstimatedPriceMC, double ErrorMC, double Tock);
		__device__ __host__ ~Output_MC_data() = default;

		__device__ __host__ void SetEstimatedPriceMC(double);
		__device__ __host__ double GetEstimatedPriceMC() const;
		__device__ __host__ void SetErrorMC(double);
		__device__ __host__ double GetErrorMC() const;
		__device__ __host__ void SetTick(double);
		__device__ __host__ double GetTick() const;

		__device__ __host__ void CompleteEvaluationOfBlackScholes(const Input_option_data&, const Input_market_data&);

		__device__ __host__ double GetBlackScholesPrice();
		__device__ __host__ double GetErrorBlackScholes();

};
#endif
