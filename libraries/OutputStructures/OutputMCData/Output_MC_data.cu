#include <cmath>	// fabs, sqrt, pow

#include "Output_MC_data.cuh"

__device__ __host__ double Output_MC_data::GetRelativeErrorEuler() const{
	return this->ErrorMCEuler / this->EstimatedPriceMCEuler;
}

__device__ __host__ double Output_MC_data::GetRelativeErrorExact() const{
	return this->ErrorMCExact / this->EstimatedPriceMCExact;
}

__device__ __host__ double Output_MC_data::GetEulerToExactDiscrepancy() const{
	return fabs((this->EstimatedPriceMCEuler - this->EstimatedPriceMCExact)/sqrt(pow(this->ErrorMCEuler,2)+pow(ErrorMCExact,2)));
}
