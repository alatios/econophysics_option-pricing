#include "Output_MC_per_thread.cuh"

using namespace std;

__device__ __host__ Output_MC_per_thread::Output_MC_per_thread(){
	m_payoffSum = 0.;
	m_squaredPayoffSum = 0.;
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

__device__ __host__ void Output_MC_per_thread::AddToPayoffSum(float payoff){
	m_payoffSum += payoff;
}

__device__ __host__ void Output_MC_per_thread::AddToSquaredPayoffSum(float payoffSquared){
	m_squaredPayoffSum += payoffSquared;
}

__device__ __host__ void Output_MC_per_thread::AddToAll(float payoff){
	this->AddToPayoffSum(payoff);
	this->AddToSquaredPayoffSum(pow(payoff,2));
}

__device__ __host__ float Output_MC_per_thread::GetPayoffSum(){
	return m_payoffSum;
}

__device__ __host__ float Output_MC_per_thread::GetSquaredPayoffSum(){
	return m_squaredPayoffSum;
}
