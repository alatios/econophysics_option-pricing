#ifndef __Output_MC_per_thread_h__
#define __Output_MC_per_thread_h__

#include <iostream>
#include <cmath>	// pow

using namespace std;

class Output_MC_per_thread{

	private:

		// Set to zero at conception
		float m_payoffSum;
		float m_squaredPayoffSum;

	public:
	
		// Constructor and destructor
		__device__ __host__ Output_MC_per_thread();
		__device__ __host__ ~Output_MC_per_thread() = default;

		// Reset sums to zero
		__device__ __host__ void ResetPayoffSum();
		__device__ __host__ void ResetSquaredPayoffSum(); 
		__device__ __host__ void ResetAllSums();
		
		// Add argument to payoffSum or squaredPayoffSum
		__device__ __host__ void AddToPayoffSum(float payoff);
		__device__ __host__ void AddToSquaredPayoffSum(float payoffSquared);
		
		// Add argument and its square to the respective sums
		__device__ __host__ void AddToAll(float payoff);
		
		// Return value of payoffSum or squaredPayoffSum
		__device__ __host__ float GetPayoffSum();
		__device__ __host__ float GetSquaredPayoffSum();

};
#endif
