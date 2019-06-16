#include "Path.cuh"

using namespace std;

__device__ __host__ Path::Path(){
	this->SetSpotPrice(0.);
	this->SetGaussianRandomVariable(0.);
}

//Constructor
__device__ __host__ Path::Path(double SpotPrice){
	this->SetSpotPrice(SpotPrice);
	this->SetGaussianRandomVariable(0.);
}
//Copy Constructor
__device__ __host__ Path::Path(const Path& p){
	this->SetGaussianRandomVariable(p.GetGaussianRandomVariable());
	this->SetSpotPrice(p.GetSpotPrice());
}

//Methods
__device__ __host__ double Path::GetSpotPrice() const{
	return _SpotPrice;
}

__device__ __host__ void Path::SetSpotPrice(double SpotPrice){
	_SpotPrice = SpotPrice;
}

__device__ __host__ double Path::GetGaussianRandomVariable() const{
	return _GaussianRandomVariable;
}

__device__ __host__ void Path::SetGaussianRandomVariable(double GaussianRandomVariable){
	_GaussianRandomVariable = GaussianRandomVariable;
}

__device__ __host__ void Path::EuleroStep(const Input_market_data& market, const Input_option_data& option){			//It takes a step according to the euler formula
	double SpotPrice_i;		//The price at the next step
	SpotPrice_i = (this->GetSpotPrice()) *
	(1
	+ market.GetRiskFreeRate() * (option.GetTimeToMaturity() / option.GetNumberOfIntervals())
	+ market.GetVolatility() * sqrt(option.GetTimeToMaturity() / option.GetNumberOfIntervals()) * (this->GetGaussianRandomVariable()));

// Geometric brownian motion, only for test purposes
/*
	SpotPrice_i = (this->GetSpotPrice()) * expf((market.GetRiskFreeRate() - 0.5 * pow(market.GetVolatility(),2)) * option.GetDeltaTime()
	+ market.GetVolatility() * this->GetGaussianRandomVariable() * sqrt(option.GetDeltaTime()));
*/

	this->SetSpotPrice(SpotPrice_i);
}
