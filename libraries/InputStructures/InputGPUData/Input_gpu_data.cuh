#ifndef __Input_gpu_data_h__
#define __Input_gpu_data_h__

#include <iostream>
#include <cmath>

#define NUMBER_OF_THREADS_PER_BLOCK 512

struct Input_gpu_data{
	unsigned int NumberOfBlocks;

	__device__ __host__ unsigned int GetNumberOfThreadsPerBlock() const;
	__device__ __host__ unsigned int GetTotalNumberOfThreads() const;

};
#endif
