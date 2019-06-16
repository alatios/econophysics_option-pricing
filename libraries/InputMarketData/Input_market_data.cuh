#ifndef __Input_Market_data_h__
#define __Input_Market_data_h__

#include <iostream>
#include <cmath>

using namespace std;

class Input_market_data{

private:
  
	double _InitialPrice;				//Initial price of the asset
	double _Volatility;					//volatility
	double _RiskFreeRate;				//risk free interest

public:

	__device__ __host__ Input_market_data(); //Default Constructor
	__device__ __host__ Input_market_data(double InitialPrice, double Volatility, double RiskFreeRate);
	__device__ __host__ Input_market_data(const Input_market_data&); //Copy constructor
	__device__ __host__ ~Input_market_data() = default;

	__device__ __host__ void SetInitialPrice(double);
	__device__ __host__ double GetInitialPrice() const;

	__device__ __host__ void SetVolatility(double);
	__device__ __host__ double GetVolatility() const;

	__device__ __host__ void SetRiskFreeRate(double);
	__device__ __host__ double GetRiskFreeRate() const;

};
#endif
