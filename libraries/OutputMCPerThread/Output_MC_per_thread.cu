#include "Output_MC_per_thread.cuh"

using namespace std;

// Constructor
__device__ __host__ Output_MC_per_thread::Output_MC_per_thread(){
	this->SetPayoffSum(0.);
	this->SetSquaredPayoffSum(0.);
	this->SetPayoffCounter(0);
	this->SetSquaredPayoffCounter(0);
}

// Private set methods
__device__ __host__ void Output_MC_per_thread::SetPayoffSum(double payoffSum){
	m_payoffSum = payoffSum;
}
__device__ __host__ void Output_MC_per_thread::SetSquaredPayoffSum(double squaredPayoffSum){
	m_squaredPayoffSum = squaredPayoffSum;
}
__device__ __host__ void Output_MC_per_thread::SetPayoffCounter(unsigned int payoffCounter){
	m_payoffCounter = payoffCounter;	
}

__device__ __host__ void Output_MC_per_thread::SetSquaredPayoffCounter(unsigned int squaredPayoffCounter){
	m_squaredPayoffCounter = squaredPayoffCounter;
}

// Private methods for counter increase
__device__ __host__ void Output_MC_per_thread::IncreasePayoffCounter(){
	++m_payoffCounter;
}

__device__ __host__ void Output_MC_per_thread::IncreaseSquaredPayoffCounter(){
	++m_squaredPayoffCounter;
}

// Private methods for counter reset
__device__ __host__ void Output_MC_per_thread::ResetPayoffCounter(){
	this->SetPayoffCounter(0);
}

__device__ __host__ void Output_MC_per_thread::ResetSquaredPayoffCounter(){
	this->SetSquaredPayoffCounter(0);
}

// Public methods for addition
__device__ __host__ void Output_MC_per_thread::AddToPayoffSum(double payoff){
	m_payoffSum += payoff;
	this->IncreasePayoffCounter();
}

__device__ __host__ void Output_MC_per_thread::AddToSquaredPayoffSum(double payoffSquared){
	m_squaredPayoffSum += payoffSquared;
	this->IncreaseSquaredPayoffCounter();
}

__device__ __host__ void Output_MC_per_thread::AddToAll(double payoff){
	this->AddToPayoffSum(payoff);
	this->AddToSquaredPayoffSum(pow(payoff,2));
}

// Public methods for sum resetting
__device__ __host__ void Output_MC_per_thread::ResetPayoffSum(){
	this->SetPayoffSum(0.);
	this->ResetPayoffCounter();
}

__device__ __host__ void Output_MC_per_thread::ResetSquaredPayoffSum(){
	this->SetSquaredPayoffSum(0.);
	this->ResetSquaredPayoffCounter();
}

__device__ __host__ void Output_MC_per_thread::ResetAllSums(){
	this->ResetPayoffSum();
	this->ResetSquaredPayoffSum();
}

// Public get methods
__device__ __host__ double Output_MC_per_thread::GetPayoffSum(){
	return m_payoffSum;
}

__device__ __host__ double Output_MC_per_thread::GetSquaredPayoffSum(){
	return m_squaredPayoffSum;
}

__device__ __host__ unsigned int Output_MC_per_thread::GetPayoffCounter(){
	return m_payoffCounter;
}

__device__ __host__ unsigned int Output_MC_per_thread::GetSquaredPayoffCounter(){
	return m_squaredPayoffCounter;
}
