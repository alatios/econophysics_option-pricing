#include "Input_gpu_data.cuh"

using namespace std;

//Default constructor
__device__ __host__ Input_gpu_data::Input_gpu_data(){         //Set the possible maximum number of blocks and threads per block on our GPUs
	this->SetNumberOfBlocks(14);
	this->SetNumberOfThreadsPerBlock(1024);
}
//Constructor
__device__ __host__ Input_gpu_data::Input_gpu_data(unsigned int NumberOfBlocks, unsigned int NumberOfThreadsPerBlock){
	this->SetNumberOfBlocks(NumberOfBlocks);
	this->SetNumberOfThreadsPerBlock(NumberOfThreadsPerBlock);
}

//Copy constructor
__device__ __host__ Input_gpu_data::Input_gpu_data(const Input_gpu_data& data){
	this->SetNumberOfBlocks(data.GetNumberOfBlocks());
	this->SetNumberOfThreadsPerBlock(data.GetNumberOfThreadsPerBlock());
}
//Methods
__device__ __host__ void Input_gpu_data::SetNumberOfBlocks(unsigned int NumberOfBlocks){
	_NumberOfBlocks = NumberOfBlocks;
}

__device__ __host__ unsigned int Input_gpu_data::GetNumberOfBlocks() const{
	return _NumberOfBlocks;
}

__device__ __host__ void Input_gpu_data::SetNumberOfThreadsPerBlock(unsigned int NumberOfThreadsPerBlock){
	_NumberOfThreadsPerBlock = NumberOfThreadsPerBlock;
}

__device__ __host__ unsigned int Input_gpu_data::GetNumberOfThreadsPerBlock() const{
	return _NumberOfThreadsPerBlock;
}

__device__ __host__ unsigned int Input_gpu_data::GetTotalNumberOfThreads() const{
	return _NumberOfThreadsPerBlock * _NumberOfBlocks;
}
