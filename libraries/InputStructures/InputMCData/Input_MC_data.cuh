#ifndef __Input_MC_data_h__
#define __Input_MC_data_h__

#include <iostream>
#include "../InputGPUData/Input_gpu_data.cuh"

struct Input_MC_data{

	unsigned int NumberOfMCSimulations;
	char CpuOrGpu;
	char GaussianOrBimodal;
	__device__ __host__ unsigned int GetNumberOfSimulationsPerThread(const Input_gpu_data&) const;

};

#endif
