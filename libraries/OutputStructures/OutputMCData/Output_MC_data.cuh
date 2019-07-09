#ifndef __Output_MC_data_h__
#define __Output_MC_data_h__

struct Output_MC_data{
	
	char HostOrDevice;
	double EstimatedPriceMCEuler;
	double ErrorMCEuler;
	double EstimatedPriceMCExact;
	double ErrorMCExact;
	double Tick;			// Calculation time [ms]
	unsigned int NegativePriceCounter;
	
	__device__ __host__ double GetRelativeErrorEuler() const;
	__device__ __host__ double GetRelativeErrorExact() const;
	__device__ __host__ double GetEulerToExactDiscrepancy() const;

};

#endif
