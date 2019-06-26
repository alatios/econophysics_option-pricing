#include "Input_option_data.cuh"

using namespace std;

__device__ __host__ double Input_option_data::GetDeltaTime() const{
	return static_cast<double>(static_cast<double>(this->TimeToMaturity) / static_cast<unsigned int>(this->NumberOfIntervals));	
}
