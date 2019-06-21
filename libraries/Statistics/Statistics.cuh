#ifndef __Statistics_h__
#define __Statistics_h__

#include <iostream>
#include <cmath>	// pow

using namespace std;

class Statistics{

	private:

		// Set to zero at conception
		double m_payoffSum;
		double m_squaredPayoffSum;
		unsigned int m_payoffCounter;
		unsigned int m_squaredPayoffCounter;
		
		// Set methods
		__device__ __host__ void SetPayoffSum(double);
		__device__ __host__ void SetSquaredPayoffSum(double);
		__device__ __host__ void SetPayoffCounter(unsigned int);
		__device__ __host__ void SetSquaredPayoffCounter(unsigned int);
		
		//Increase counters
		__device__ __host__ void IncreasePayoffCounter();
		__device__ __host__ void IncreaseSquaredPayoffCounter();
		
		// Reset counters to zero
		__device__ __host__ void ResetPayoffCounter();
		__device__ __host__ void ResetSquaredPayoffCounter();

	public:
	
		// Constructor and destructor
		__device__ __host__ Statistics();
		__device__ __host__ ~Statistics() = default;

		// Add argument to payoffSum or squaredPayoffSum
		__device__ __host__ void AddToPayoffSum(double payoff);
		__device__ __host__ void AddToSquaredPayoffSum(double payoffSquared);
	
		// Add argument and its square to the respective sums
		__device__ __host__ void AddToAll(double payoff);
		
		// Reset sums (and respective counters)
		__device__ __host__ void ResetPayoffSum();
		__device__ __host__ void ResetSquaredPayoffSum(); 
		__device__ __host__ void ResetAllSums();
				
		// Return value of payoffSum or squaredPayoffSum (and respective counters)
		__device__ __host__ double GetPayoffSum();
		__device__ __host__ double GetSquaredPayoffSum();
		__device__ __host__ unsigned int GetPayoffCounter();
		__device__ __host__ unsigned int GetSquaredPayoffCounter();
};
#endif
