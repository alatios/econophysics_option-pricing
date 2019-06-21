#include "Statistics.cuh"

using namespace std;

// Constructor
__device__ __host__ Statistics::Statistics(){
	this->SetPayoffSum(0.);
	this->SetSquaredPayoffSum(0.);
	this->SetPayoffCounter(0);
	this->SetSquaredPayoffCounter(0);
}

// Private set methods
__device__ __host__ void Statistics::SetPayoffSum(double payoffSum){
	m_payoffSum = payoffSum;
}
__device__ __host__ void Statistics::SetSquaredPayoffSum(double squaredPayoffSum){
	m_squaredPayoffSum = squaredPayoffSum;
}
__device__ __host__ void Statistics::SetPayoffCounter(unsigned int payoffCounter){
	m_payoffCounter = payoffCounter;	
}

__device__ __host__ void Statistics::SetSquaredPayoffCounter(unsigned int squaredPayoffCounter){
	m_squaredPayoffCounter = squaredPayoffCounter;
}

// Private methods for counter increase
__device__ __host__ void Statistics::IncreasePayoffCounter(){
	++m_payoffCounter;
}

__device__ __host__ void Statistics::IncreaseSquaredPayoffCounter(){
	++m_squaredPayoffCounter;
}

// Private methods for counter reset
__device__ __host__ void Statistics::ResetPayoffCounter(){
	this->SetPayoffCounter(0);
}

__device__ __host__ void Statistics::ResetSquaredPayoffCounter(){
	this->SetSquaredPayoffCounter(0);
}

// Public methods for addition
__device__ __host__ void Statistics::AddToPayoffSum(double payoff){
	m_payoffSum += payoff;
	this->IncreasePayoffCounter();
}

__device__ __host__ void Statistics::AddToSquaredPayoffSum(double payoffSquared){
	m_squaredPayoffSum += payoffSquared;
	this->IncreaseSquaredPayoffCounter();
}

__device__ __host__ void Statistics::AddToAll(double payoff){
	this->AddToPayoffSum(payoff);
	this->AddToSquaredPayoffSum(pow(payoff,2));
}

// Public methods for sum resetting
__device__ __host__ void Statistics::ResetPayoffSum(){
	this->SetPayoffSum(0.);
	this->ResetPayoffCounter();
}

__device__ __host__ void Statistics::ResetSquaredPayoffSum(){
	this->SetSquaredPayoffSum(0.);
	this->ResetSquaredPayoffCounter();
}

__device__ __host__ void Statistics::ResetAllSums(){
	this->ResetPayoffSum();
	this->ResetSquaredPayoffSum();
}

// Public get methods
__device__ __host__ double Statistics::GetPayoffSum(){
	return m_payoffSum;
}

__device__ __host__ double Statistics::GetSquaredPayoffSum(){
	return m_squaredPayoffSum;
}

__device__ __host__ unsigned int Statistics::GetPayoffCounter(){
	return m_payoffCounter;
}

__device__ __host__ unsigned int Statistics::GetSquaredPayoffCounter(){
	return m_squaredPayoffCounter;
}
