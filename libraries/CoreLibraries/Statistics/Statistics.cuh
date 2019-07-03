#ifndef __Statistics_h__
#define __Statistics_h__

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
		__device__ __host__ double GetPayoffSum() const;
		__device__ __host__ double GetSquaredPayoffSum() const;
		__device__ __host__ unsigned int GetPayoffCounter() const;
		
		// Evaluate average and error
		__device__ __host__ void EvaluateEstimatedPriceAndError();
		__device__ __host__ double GetPayoffAverage() const;
		__device__ __host__ double GetPayoffError() const;
		
		// Overload of += operator
		__host__ Statistics& operator+=(const Statistics&);	// Host-only because of isinf, at least for now

};

#endif
