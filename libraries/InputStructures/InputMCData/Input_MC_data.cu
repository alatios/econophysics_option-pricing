#include <cmath>	// ceil
#include "Input_MC_data.cuh"

using namespace std;

__device__ __host__ unsigned int Input_MC_data::GetNumberOfSimulationsPerThread(const Input_gpu_data& inputGPU) const{
	return ceil(static_cast<double>(this->NumberOfMCSimulations / inputGPU.GetTotalNumberOfThreads());
}
