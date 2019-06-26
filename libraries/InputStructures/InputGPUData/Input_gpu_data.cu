#include "Input_gpu_data.cuh"

using namespace std;

__device__ __host__ unsigned int Input_gpu_data::GetNumberOfThreadsPerBlock() const{
	return NUMBER_OF_THREADS_PER_BLOCK;
}

__device__ __host__ unsigned int Input_gpu_data::GetTotalNumberOfThreads() const{
	return this->_NumberOfBlocks * NUMBER_OF_THREADS_PER_BLOCK;
}
