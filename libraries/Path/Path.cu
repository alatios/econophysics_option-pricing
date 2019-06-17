#include "Path.cuh"

using namespace std;

// Constructors
__device__ __host__ Path::Path(){
	this->SetSpotPrice(0.);
	this->SetRiskFreeRate(0.1);
	this->SetVolatility(0.25);
	this->SetDeltaTime(1./365);
}

__device__ __host__ Path::Path(const Input_market_data& market, const Input_option_data& option, double SpotPrice){
	this->SetSpotPrice(SpotPrice);
	this->SetRiskFreeRate(market.GetRiskFreeRate());
	this->SetVolatility(market.GetVolatility());
	this->SetDeltaTime(option.GetDeltaTime());
}

// Private set methods
__device__ __host__ void Path::SetSpotPrice(double spotPrice){
	_SpotPrice = spotPrice;
}

__device__ __host__ void Path::SetRiskFreeRate(double riskFreeRate){
	_RiskFreeRate = riskFreeRate;
}

__device__ __host__ void Path::SetVolatility(double volatility){
	_Volatility = volatility;
}

__device__ __host__ void Path::SetDeltaTime(double deltaTime){
	_DeltaTime = deltaTime;
}

// Public set methods
__device__ __host__ SetInternalState(const Input_market_data& market, const Input_option_data& option, double SpotPrice){
	this->SetSpotPrice(SpotPrice);
	this->SetRiskFreeRate(market.GetRiskFreeRate());
	this->SetVolatility(market.GetVolatility());
	this->SetDeltaTime(option.GetDeltaTime());	
}

__device__ __host__ SetInternalState(const Path& otherPath){
	this->SetSpotPrice(otherPath.GetSpotPrice());
	this->SetRiskFreeRate(otherPath.GetRiskFreeRate());
	this->SetVolatility(otherPath.GetVolatility());
	this->SetDeltaTime(otherPath.GetDeltaTime());		
}

// Public get methods
__device__ __host__ double Path::GetSpotPrice() const{
	return _SpotPrice;
}

__device__ __host__ double Path::GetVolatility() const{
	return _Volatility;
}
__device__ __host__ double Path::GetRiskFreeRate() const{
	return _RiskFreeRate;
}
__device__ __host__ double Path::GetDeltaTime() const{
	return _DeltaTime;
}


// Euler step implementation
__device__ __host__ void Path::EuleroStep(double gaussianRandomVariable){
	double SpotPrice_i;		//The price at the next step
	SpotPrice_i = (this->GetSpotPrice()) *
	(1 + this->GetRiskFreeRate() * this->GetDeltaTime()
	+ this->GetVolatility() * sqrt(this->GetDeltaTime()) * gaussianRandomVariable);

// Geometric brownian motion, only for test purposes
/*
	SpotPrice_i = (this->GetSpotPrice()) * expf((market.GetRiskFreeRate() - 0.5 * pow(market.GetVolatility(),2)) * option.GetDeltaTime()
	+ market.GetVolatility() * this->GetGaussianRandomVariable() * sqrt(option.GetDeltaTime()));
*/

	this->SetSpotPrice(SpotPrice_i);
}
