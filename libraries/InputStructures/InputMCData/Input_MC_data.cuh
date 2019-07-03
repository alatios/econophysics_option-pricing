#ifndef __Input_MC_data_h__
#define __Input_MC_data_h__

#include "../InputGPUData/Input_gpu_data.cuh"

struct Input_MC_data{

	unsigned int NumberOfMCSimulations;
	char CpuOrGpu;	// g = gpu algorithm only
					// c = cpu algorithm only
					// b = both algorithms are used
	char GaussianOrBimodal;	// g = gaussian variables are used
							// b = bimodal (1/-1) variables are used
	
	__device__ __host__ unsigned int GetNumberOfSimulationsPerThread(const Input_gpu_data&) const;

};

#endif
