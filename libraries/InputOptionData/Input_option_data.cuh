#ifndef __Input_option_data_h__
#define __Input_option_data_h__

#include <iostream>
#include <cmath>

using namespace std;

class Input_option_data{

private:

	char _OptionType;
	double _StrikePrice;
	unsigned int _NumberOfIntervals;
	double _TimeToMaturity;				//Time passed from the inizial istant [days]

public:

	__device__ __host__ Input_option_data(); //Default constructor
	__device__ __host__ Input_option_data(double, unsigned int, double, char);
	__device__ __host__ Input_option_data(const Input_option_data&); //Copy constructor
	__device__ __host__ ~Input_option_data() = default;

	__device__ __host__ void SetStrikePrice(double);
	__device__ __host__ double GetStrikePrice() const;
	__device__ __host__ void SetNumberOfIntervals(unsigned int);
	__device__ __host__ unsigned int GetNumberOfIntervals() const;
	__device__ __host__ void SetTimeToMaturity(double);
	__device__ __host__ double GetTimeToMaturity() const;
	__device__ __host__ void SetOptionType(const char);
	__device__ __host__ char GetOptionType() const;


};
#endif
