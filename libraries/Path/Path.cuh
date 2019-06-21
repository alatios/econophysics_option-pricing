#ifndef __Path_h__
#define __Path_h__

#include "../InputMarketData/Input_market_data.cuh"
#include "../InputOptionData/Input_option_data.cuh"

#include <iostream>
#include <cmath>

using namespace std;


class Path{

	private:

		double _SpotPrice;		// The step (spotprice) required to generate the next one
		double _RiskFreeRate;
		double _Volatility;
		double _DeltaTime;
		
		__device__ __host__ void SetSpotPrice(double);
		__device__ __host__ void SetRiskFreeRate(double);
		__device__ __host__ void SetVolatility(double);
		__device__ __host__ void SetDeltaTime(double);

		__device__ __host__ double GetVolatility() const;
		__device__ __host__ double GetRiskFreeRate() const;
		__device__ __host__ double GetDeltaTime() const;

	public:

		__device__ __host__ Path();
		__device__ __host__ Path(const Input_market_data& market, const Input_option_data& option, double SpotPrice);
		__device__ __host__ ~Path() = default;

		__device__ __host__ void SetInternalState(const Input_market_data& market, const Input_option_data& option, double SpotPrice);
		__device__ __host__ void SetInternalState(const Path&);

		__device__ __host__ void EuleroStep(double gaussianRandomVariable);
		__device__ __host__ double GetSpotPrice() const;
};
#endif
