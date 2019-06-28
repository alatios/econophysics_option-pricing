#ifndef __Path_h__
#define __Path_h__

#include "../../InputStructures/InputMarketData/Input_market_data.cuh"
#include "../../InputStructures/InputOptionData/Input_option_data.cuh"

#include <iostream>

class Path{

	private:

		// Ricordarsi di mettere i puntatori
		
		double _SpotPrice;		// The step (spotprice) required to generate the next one

		// Market data
		double _RiskFreeRate;
		double _Volatility;

		// Base European option/contract data
		char _OptionType;
		unsigned int _TimeToMaturity;
		unsigned int _NumberOfIntervals;
		double _DeltaTime;
		
		// Plain vanilla option data
		double _StrikePrice;
		
		// Performance corridor data
		double _B;
		double _N;
		double _K;
		unsigned int _PerformanceCorridorBarrierCounter;
		
		__device__ __host__ void CheckPerformanceCorridorCondition(double currentSpotPrice, double nextSpotPrice);
		
	public:

		__device__ __host__ Path() = default;
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
};

#endif
