#include "Output_MC_per_thread.cuh"

using namespace std;

__device__ __host__ Output_MC_per_thread::Output_MC_per_thread(){
	m_payoffSum = 0.;
	m_squaredPayoffSum = 0.;
	m_cumulativePayoffs = 0;
	m_cumulativeSquaredPayoffs = 0;
}

__device__ __host__ void Output_MC_per_thread::ResetPayoffSum(){
	m_payoffSum = 0.;
}

__device__ __host__ void Output_MC_per_thread::ResetSquaredPayoffSum(){
	m_squaredPayoffSum = 0.;
}
__device__ __host__ void Output_MC_per_thread::ResetAllSums(){
	this->ResetPayoffSum();
	this->ResetSquaredPayoffSum();
}

__device__ __host__ void Output_MC_per_thread::ResetCumulativePayoffs(){
	m_cumulativePayoffs = 0;
}

__device__ __host__ void Output_MC_per_thread::ResetCumulativeSquaredPayoffs(){
	m_cumulativeSquaredPayoffs = 0;
}

__device__ __host__ void Output_MC_per_thread::AddToPayoffSum(double payoff){
	m_payoffSum += payoff;
	this->IncreaseCumulativePayoffs();
}

__device__ __host__ void Output_MC_per_thread::AddToSquaredPayoffSum(double payoffSquared){
	m_squaredPayoffSum += payoffSquared;
	this->IncreaseCumulativeSquaredPayoffs();
}
		
__device__ __host__ void Output_MC_per_thread::IncreaseCumulativePayoffs(){
	m_cumulativePayoffs += 1;
}

__device__ __host__ void Output_MC_per_thread::IncreaseCumulativeSquaredPayoffs(){
	m_cumulativeSquaredPayoffs += 1;
}

__device__ __host__ void Output_MC_per_thread::AddToAll(double payoff){
	this->AddToPayoffSum(payoff);
	this->AddToSquaredPayoffSum(pow(payoff,2));
}

__device__ __host__ double Output_MC_per_thread::GetPayoffSum(){
	return m_payoffSum;
}

__device__ __host__ double Output_MC_per_thread::GetSquaredPayoffSum(){
	return m_squaredPayoffSum;
}

__device__ __host__ double Output_MC_per_thread::GetCumulativePayoffs(){
	return m_cumulativePayoffs;
}

__device__ __host__ double Output_MC_per_thread::GetCumulativeSquaredPayoffs(){
	return m_cumulativePayoffs;
}