#ifndef __Input_MC_data_h__
#define __Input_MC_data_h__

#include <iostream>
#include <cmath>
#include "../InputGPUData/Input_gpu_data.cuh"

using namespace std;

class Input_MC_data{
	private:
	
		Input_gpu_data _GpuData;
		unsigned int _NumberOfMCSimulationsPerThread;
		unsigned int _NumberOfMCSimulations;

	public:

		__device__ __host__ Input_MC_data();
		__device__ __host__ Input_MC_data(unsigned int);
		__device__ __host__ Input_MC_data(Input_gpu_data);
		__device__ __host__ Input_MC_data(unsigned int, Input_gpu_data);
		__device__ __host__ ~Input_MC_data() = default;

		__device__ __host__ void SetNumberOfMCSimulations();
		__device__ __host__ unsigned int GetNumberOfMCSimulations();

		__device__ __host__ void SetNumberOfMCSimulationsPerThread(unsigned int);
		__device__ __host__ unsigned int GetNumberOfMCSimulationsPerThread() const;
		
		__device__ __host__ void SetGpuData(Input_gpu_data);
		__device__ __host__ Input_gpu_data GetGpuData();

};
#endif
