#ifndef __Input_option_data_h__
#define __Input_option_data_h__

#include <iostream>
#include <cmath>

using namespace std;

struct Input_option_data{

	// fw = forward contract
	// vc = plain vanilla call option
	// vp = plain vanilla put option
	// pc = performance corridor option
	char _OptionType[2];
	unsigned int _NumberOfIntervals;
	double _TimeToMaturity;				// Time passed from the initial istant [years]

	__device__ __host__ double GetDeltaTime() const;

};

struct Input_option_data_PlainVanilla: public Input_option_data{
	
	double _StrikePrice;

};

struct Input_option_data_PerformanceCorridor: public Input_option_data{
	
	double _StrikePrice;
	double _B;
	double _K;
	double _N;

};
#endif
