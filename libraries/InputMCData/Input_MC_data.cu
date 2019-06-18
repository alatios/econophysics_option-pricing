#include "Input_MC_data.cuh"

using namespace std;

//Constructor
__device__ __host__ Input_MC_data::Input_MC_data(){
	this->SetNumberOfMCSimulations(5000000);
}

__device__ __host__ Input_MC_data::Input_MC_data(unsigned int NumberOfMCSimulations){
	this->SetNumberOfMCSimulations(NumberOfMCSimulations);
}

//Methods
__device__ __host__ void Input_MC_data::SetNumberOfMCSimulations(unsigned int NumberOfMCSimulations){
	_NumberOfMCSimulations = static_cast<unsigned int>(NumberOfMCSimulations);
}

__device__ __host__ unsigned int Input_MC_data::GetNumberOfMCSimulations() const{
	return _NumberOfMCSimulations;
}

__device__ __host__ unsigned int Input_MC_data::GetNumberOfSimulationsPerThread(const Input_gpu_data& inputGPU) const{
	return ceil(static_cast<double>(this->GetNumberOfMCSimulations()) / inputGPU.GetTotalNumberOfThreads());
}
