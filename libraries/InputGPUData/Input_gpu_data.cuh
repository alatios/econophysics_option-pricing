#ifndef __Input_gpu_data_h__
#define __Input_gpu_data_h__

#include <iostream>
#include <cmath>

using namespace std;

class Input_gpu_data{
private:

	unsigned int _NumberOfBlocks;
	unsigned int _NumberOfThreadsPerBlock;

public:

	__device__ __host__ Input_gpu_data();
	__device__ __host__ Input_gpu_data(unsigned int NumberOfBlocks, unsigned int NumberOfThreadsPerBlock);
	__device__ __host__ Input_gpu_data(const Input_gpu_data&);
	__device__ __host__ ~Input_gpu_data() = default;

	__device__ __host__ void SetNumberOfBlocks(unsigned int);
	__device__ __host__ unsigned int GetNumberOfBlocks() const;
	__device__ __host__ void SetNumberOfThreadsPerBlock(unsigned int);
	__device__ __host__ unsigned int GetNumberOfThreadsPerBlock() const;
	__device__ __host__ unsigned int GetTotalNumberOfThreads() const;
};
#endif
