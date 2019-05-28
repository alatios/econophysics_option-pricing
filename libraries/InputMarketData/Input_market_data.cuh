#ifndef __Input_Market_data_h__
#define __Input_Market_data_h__

#include <iostream>
#include <cmath>

using namespace std;

class Input_market_data{

private:
  
	float _ZeroPrice;					//Initial price of the asset
	float _Volatility;					//volatility
	float _RiskFreeRate;				//risk free interest

public:

	__device__ __host__ Input_market_data(); //Default Constructor
	__device__ __host__ Input_market_data(float ZeroPrice, float Volatility, float RiskFreeRate);
	__device__ __host__ Input_market_data(const Input_market_data&); //Copy constructor
	__device__ __host__ ~Input_market_data() = default;

	__device__ __host__ void SetZeroPrice(float);
	__device__ __host__ float GetZeroPrice() const;

	__device__ __host__ void SetVolatility(float);
	__device__ __host__ float GetVolatility() const;

	__device__ __host__ void SetRiskFreeRate(float);
	__device__ __host__ float GetRiskFreeRate() const;

};
#endif
