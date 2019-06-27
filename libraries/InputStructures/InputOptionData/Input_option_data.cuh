#ifndef __Input_option_data_h__
#define __Input_option_data_h__

#include <iostream>
#include <cmath>

using namespace std;

struct Input_option_data{
	
	// Common to all option types


	char OptionType;		// f = forward contract
							// c = plain vanilla call option
							// p = plain vanilla put option
							// e = performance corridor option
	unsigned int NumberOfIntervals;
	double TimeToMaturity;

	__device__ __host__ double GetDeltaTime() const;
	
	// Specific to plain vanilla options
	double StrikePrice;
		
	// Specific to performance corridor options
	double B;
	double K;
	double N;

};

#endif
