#ifndef __Input_option_data_h__
#define __Input_option_data_h__

#include <iostream>
#include <cmath>

using namespace std;

class Input_option_data{

private:

	char _OptionType;
	float _StrikePrice;
	unsigned int _NumberOfIntervals;
	float _TimeToMaturity;				//Time passed from the inizial istant [days]
	float _DeltaTime;					//Time steps in which the time to maturity is divided

public:

	__device__ __host__ Input_option_data(); //Default constructor
	__device__ __host__ Input_option_data(float, unsigned int, float, char);
	__device__ __host__ Input_option_data(const Input_option_data&); //Copy constructor
	__device__ __host__ ~Input_option_data() = default;

	__device__ __host__ void SetStrikePrice(float);
	__device__ __host__ float GetStrikePrice() const;
	__device__ __host__ void SetNumberOfIntervals(unsigned int);
	__device__ __host__ unsigned int GetNumberOfIntervals() const;
	__device__ __host__ void SetTimeToMaturity(float);
	__device__ __host__ float GetTimeToMaturity() const;
	__device__ __host__ void SetDeltaTime();
	__device__ __host__ float GetDeltaTime() const;
	__device__ __host__ void SetOptionType(const char);
	__device__ __host__ char GetOptionType() const;


};
#endif
