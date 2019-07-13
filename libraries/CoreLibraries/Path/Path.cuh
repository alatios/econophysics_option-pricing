#ifndef __Path_h__
#define __Path_h__

#include "../../InputStructures/InputMarketData/Input_market_data.cuh"
#include "../../InputStructures/InputOptionData/Input_option_data.cuh"

class Path{

	private:
		
		double _SpotPrice;		// The step (spotprice) required to generate the next one

		// Market data
		const double* _RiskFreeRate;
		const double* _Volatility;
		const double* _InitialPrice;

		// Base European option/contract data
		const char* _OptionType;
		const unsigned int* _NumberOfIntervals;
		const double* _TimeToMaturity;
		double _DeltaTime;
		
		// Plain vanilla option data
		const double* _StrikePrice;
		
		// Performance corridor data
		const double* _B;
		const double* _K;
		const double* _N;
		unsigned int _PerformanceCorridorBarrierCounter;
		
		// Boolean keeping track of negative prices in the Euler formula
		bool _NegativePrice;
		
		__device__ __host__ void CheckPerformanceCorridorCondition(double currentSpotPrice, double nextSpotPrice);
		
	public:

		__device__ __host__ Path();
		__device__ __host__ Path(const Input_market_data& market, const Input_option_data& option);
		__device__ __host__ ~Path() = default;

		__device__ __host__ void ResetToInitialState(const Input_market_data& market, const Input_option_data& option);
		__device__ __host__ void ResetToInitialState(const Path&);

		__device__ __host__ void EulerLogNormalStep(double gaussianRandomVariable);
		__device__ __host__ void ExactLogNormalStep(double gaussianRandomVariable);
		
		__device__ __host__ double GetSpotPrice() const;
		__device__ __host__ unsigned int GetPerformanceCorridorBarrierCounter() const;
		
		// Payoff evaluation
		__device__ __host__ double GetActualizedPayoff() const;
		
		// Check if a negative price happened in this run
		__device__ __host__ bool GetNegativePrice() const;

		// Black & Scholes formula
		__device__ __host__ double GetBlackAndScholesPrice() const;

};

#endif
