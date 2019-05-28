#include "Input_MC_data.cuh"

using namespace std;


//Constructor
__device__ __host__ Input_MC_data::Input_MC_data(){
	this->SetNumberOfMCSimulationsPerThread(5);
	Input_gpu_data GpuData;
	this->SetGpuData(GpuData);
	this->SetNumberOfMCSimulations();
}

__device__ __host__ Input_MC_data::Input_MC_data(unsigned int NumberOfMCSimulationsPerThread){
	this->SetNumberOfMCSimulationsPerThread(NumberOfMCSimulationsPerThread);
	Input_gpu_data GpuData;
	this->SetGpuData(GpuData);
	this->SetNumberOfMCSimulations();
}

__device__ __host__ Input_MC_data::Input_MC_data(Input_gpu_data GpuData){
	this->SetNumberOfMCSimulationsPerThread(5);
	this->SetGpuData(GpuData);
	this->SetNumberOfMCSimulations();
}

__device__ __host__ Input_MC_data::Input_MC_data(unsigned int NumberOfMCSimulationsPerThread, Input_gpu_data GpuData){
	this->SetNumberOfMCSimulationsPerThread(NumberOfMCSimulationsPerThread);
	this->SetGpuData(GpuData);
	this->SetNumberOfMCSimulations();
}

//Methods
__device__ __host__ void Input_MC_data::SetNumberOfMCSimulations(){
	_NumberOfMCSimulations = static_cast<unsigned int>(this->GetNumberOfMCSimulationsPerThread()*this->GetGpuData().GetTotalNumberOfThreads());
}

__device__ __host__ unsigned int Input_MC_data::GetNumberOfMCSimulations(){
	this->SetNumberOfMCSimulations();
	return _NumberOfMCSimulations;
}

__device__ __host__ void Input_MC_data::SetNumberOfMCSimulationsPerThread(unsigned int NumberOfMCSimulationsPerThread){
	_NumberOfMCSimulationsPerThread = NumberOfMCSimulationsPerThread;
	this->SetNumberOfMCSimulations();
}
__device__ __host__ unsigned int Input_MC_data::GetNumberOfMCSimulationsPerThread() const{
	return _NumberOfMCSimulationsPerThread;
}

__device__ __host__ void Input_MC_data::SetGpuData(Input_gpu_data GpuData){
	_GpuData.SetNumberOfThreadsPerBlock(GpuData.GetNumberOfThreadsPerBlock());
	_GpuData.SetNumberOfBlocks(GpuData.GetNumberOfBlocks());
	this->SetNumberOfMCSimulations();
}

__device__ __host__ Input_gpu_data Input_MC_data::GetGpuData(){
	return _GpuData;
}
