#include <cmath>	// pow, sqrt
#include "Statistics.cuh"

using namespace std;

// Constructor
__device__ __host__ Statistics::Statistics(){
	this->_PayoffSum = 0.;
	this->_SquaredPayoffSum = 0.;
	this->_PayoffCounter = 0;
}

// Public methods for addition
__device__ __host__ void Statistics::AddPayoff(double payoff){
	this->_PayoffSum += payoff;
	this->_SquaredPayoffSum += pow(payoff,2);
}

// Public methods for sum resetting
__device__ __host__ void Statistics::ResetSums(){
	this->_PayoffSum = 0.;
	this->_SquaredPayoffSum = 0.;
	this->_PayoffCounter = 0;
}

// Public get methods
__device__ __host__ double Statistics::GetPayoffSum(){
	return this->_PayoffSum;
}

__device__ __host__ double Statistics::GetSquaredPayoffSum(){
	return this->_SquaredPayoffSum;
}

__device__ __host__ unsigned int Statistics::GetPayoffCounter(){
	return this->_PayoffCounter;
}

// Average and error evaluation and output
__device__ __host__ void Statistics::EvaluateEstimatedPriceAndError(){
	this->_PayoffAverage = this->_PayoffSum / this->_PayoffCounter;
	this->_PayoffError = sqrt((this->_SquaredPayoffSum / this->_PayoffCounter - pow(this->_PayoffAverage,2))/ this->_PayoffCounter);
}

__device__ __host__ double Statistics::GetPayoffAverage(){
	return this->_PayoffAverage;
}


__device__ __host__ double Statistics::GetPayoffError(){
	return this->_PayoffError;
}

// Operator+= overload
__device__ __host__ Statistics& Statistics::operator+=(const Statistics& otherStatistics){

	if(std::isinf(otherStatistics.GetPayoffSum()) || std::isinf(otherStatistics.GetSquaredPayoffSum())){
		this->_PayoffSum += otherStatistics.GetPayoffSum();
		this->_SquaredPayoffSum += otherStatistics.GetSquaredPayoffSum();
		this->_PayoffCounter += otherStatistics.GetPayoffCounter();
	}
	
	return *this;
}



