#ifndef __Input_gpu_data_h__
#define __Input_gpu_data_h__

#include <iostream>
#include <cmath>

#define NUMBER_OF_THREADS_PER_BLOCK 512

using namespace std;

class Input_gpu_data{
private:

	unsigned int _NumberOfBlocks;

public:

	__device__ __host__ Input_gpu_data();
	__device__ __host__ Input_gpu_data(unsigned int NumberOfBlocks);
	__device__ __host__ Input_gpu_data(const Input_gpu_data&);
	__device__ __host__ ~Input_gpu_data() = default;

	__device__ __host__ void SetNumberOfBlocks(unsigned int);
	__device__ __host__ unsigned int GetNumberOfBlocks() const;
	
	__device__ __host__ unsigned int GetNumberOfThreadsPerBlock() const;
	__device__ __host__ unsigned int GetTotalNumberOfThreads() const;
	
	__host__ void PrintGPUInput() const;

};
#endif
