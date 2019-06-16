#ifndef __Path_h__
#define __Path_h__

#include "../InputMarketData/Input_market_data.cuh"
#include "../InputOptionData/Input_option_data.cuh"

#include <iostream>
#include <cmath>

using namespace std;


class Path{

private:

	double _GaussianRandomVariable;					//Mean = 0; Variance = 1
	double _SpotPrice;								//The step (spotprice) required to generate the next one

public:

	__device__ __host__ Path();
	__device__ __host__ Path(double SpotPrice);
	__device__ __host__ Path(const Path&);
	__device__ __host__ ~Path() = default;

	__device__ __host__ double GetGaussianRandomVariable() const;
	__device__ __host__ void SetGaussianRandomVariable(double);
	__device__ __host__ double GetSpotPrice() const;
	__device__ __host__ void SetSpotPrice(double);

	__device__ __host__ void EuleroStep(const Input_market_data& market, const Input_option_data& option);
};
#endif
