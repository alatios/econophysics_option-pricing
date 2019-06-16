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