#include "Input_gpu_data.cuh"

using namespace std;

//Default constructor
__device__ __host__ Input_gpu_data::Input_gpu_data(){         //Set the possible maximum number of blocks on our GPUs
	this->SetNumberOfBlocks(14);
}
//Constructor
__device__ __host__ Input_gpu_data::Input_gpu_data(unsigned int NumberOfBlocks){
	this->SetNumberOfBlocks(NumberOfBlocks);
}
//Copy constructor
__device__ __host__ Input_gpu_data::Input_gpu_data(const Input_gpu_data& data){
	this->SetNumberOfBlocks(data.GetNumberOfBlocks());
}
//Methods
__device__ __host__ void Input_gpu_data::SetNumberOfBlocks(unsigned int NumberOfBlocks){
	_NumberOfBlocks = NumberOfBlocks;
}

__device__ __host__ unsigned int Input_gpu_data::GetNumberOfBlocks() const{
	return _NumberOfBlocks;
}

__device__ __host__ unsigned int Input_gpu_data::GetNumberOfThreadsPerBlock() const{
	return NUMBER_OF_THREADS_PER_BLOCK;
}

__device__ __host__ unsigned int Input_gpu_data::GetTotalNumberOfThreads() const{
	return this->GetNumberOfBlocks() * NUMBER_OF_THREADS_PER_BLOCK;
}

__host__ void Input_gpu_data::PrintGPUInput() const{
	cout << "Number of blocks: " << this->GetNumberOfBlocks() << endl;
	cout << "Number of threads per block: " << this->GetNumberOfThreadsPerBlock() << endl;
	cout << "Total number of threads: " << this->GetTotalNumberOfThreads() << endl;	
}
