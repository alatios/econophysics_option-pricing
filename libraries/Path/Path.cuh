#ifndef __Path_h__
#define __Path_h__

#include "../InputMarketData/Input_market_data.cuh"
#include "../InputOptionData/Input_option_data.cuh"

#include <iostream>
#include <cmath>

using namespace std;


class Path{

private:

	Input_market_data _MarketData;
	Input_option_data _OptionData;
	float _GaussianRandomVariable;					//Mean = 0; Variance = 1
	float _SpotPrice;								//The step (spotprice) required to generate the next one

public:

	__device__ __host__ Path();
	__device__ __host__ Path(const Input_market_data&, const Input_option_data&, float SpotPrice);
	__device__ __host__ Path(const Path&);
	__device__ __host__ ~Path() = default;

	__device__ __host__ float GetGaussianRandomVariable() const;
	__device__ __host__ void SetGaussianRandomVariable(float);
	__device__ __host__ float GetSpotPrice() const;
	__device__ __host__ void SetSpotPrice(float);
	
	__device__ __host__ void SetInputMarketData(const Input_market_data&);
	__device__ __host__ Input_market_data GetInputMarketData() const;
	__device__ __host__ void SetInputOptionData(const Input_option_data&);
	__device__ __host__ Input_option_data GetInputOptionData() const;

	__device__ __host__ void EuleroStep();
};
#endif
