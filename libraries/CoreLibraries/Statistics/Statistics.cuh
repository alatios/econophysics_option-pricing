#ifndef __Statistics_h__
#define __Statistics_h__

#include <iostream>

class Statistics{

	private:

		double _PayoffSum;
		double _SquaredPayoffSum;
		unsigned int _PayoffCounter;
		
		double _PayoffAverage;
		double _PayoffError;
		
	public:
	
		// Constructor and destructor
		__device__ __host__ Statistics();
		__device__ __host__ ~Statistics() = default;

		// Add argument and its square to the respective sums
		__device__ __host__ void AddPayoff(double payoff);
		
		// Reset sums and counter
		__device__ __host__ void ResetSums();
				
		// Return value of payoffSum or squaredPayoffSum (and respective counters)
		__device__ __host__ double GetPayoffSum();
		__device__ __host__ double GetSquaredPayoffSum();
		__device__ __host__ unsigned int GetPayoffCounter();
		
		// Evaluate average and error
		__device__ __host__ void EvaluateEstimatedPriceAndError();
		__device__ __host__ double GetPayoffAverage();
		__device__ __host__ double GetPayoffError();
		
		// Overload of += operator
		__device__ __host__ Statistics& operator+=(const Statistics&);

};

#endif
