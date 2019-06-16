#ifndef __Output_MC_per_thread_h__
#define __Output_MC_per_thread_h__

#include <iostream>
#include <cmath>	// pow

using namespace std;

class Output_MC_per_thread{

	private:

		// Set to zero at conception
		double m_payoffSum;
		double m_squaredPayoffSum;
		unsigned int m_cumulativePayoffs;
		unsigned int m_cumulativeSquaredPayoffs;

	public:
	
		// Constructor and destructor
		__device__ __host__ Output_MC_per_thread();
		__device__ __host__ ~Output_MC_per_thread() = default;

		// Reset sums to zero
		__device__ __host__ void ResetPayoffSum();
		__device__ __host__ void ResetSquaredPayoffSum(); 
		__device__ __host__ void ResetAllSums();
		__device__ __host__ void ResetCumulativePayoffs();
		__device__ __host__ void ResetCumulativeSquaredPayoffs();

		// Add argument to payoffSum or squaredPayoffSum
		__device__ __host__ void AddToPayoffSum(double payoff);
		__device__ __host__ void AddToSquaredPayoffSum(double payoffSquared);

		//Increase the addend of payoffSum or squaredPayoffSum
		__device__ __host__ void IncreaseCumulativePayoffs();
		__device__ __host__ void IncreaseCumulativeSquaredPayoffs();
		
		// Add argument and its square to the respective sums
		__device__ __host__ void AddToAll(double payoff);
		
		// Return value of payoffSum or squaredPayoffSum
		__device__ __host__ double GetPayoffSum();
		__device__ __host__ double GetSquaredPayoffSum();
		__device__ __host__ double GetCumulativePayoffs();
		__device__ __host__ double GetCumulativeSquaredPayoffs();
};
#endif
