#ifndef __Input_MC_data_h__
#define __Input_MC_data_h__

#include <iostream>
#include <cmath>

using namespace std;

class Input_MC_data{
	private:

		unsigned int _NumberOfMCSimulations;

	public:

		__device__ __host__ Input_MC_data();
		__device__ __host__ Input_MC_data(unsigned int);
		__device__ __host__ ~Input_MC_data() = default;

		__device__ __host__ void SetNumberOfMCSimulations(unsigned int);
		__device__ __host__ unsigned int GetNumberOfMCSimulations() const;
};

#endif
