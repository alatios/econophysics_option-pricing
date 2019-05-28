#include "Input_market_data.cuh"

using namespace std;


//Default Constructor
__device__ __host__ Input_market_data::Input_market_data(){
	this->SetZeroPrice(100.);
	this->SetVolatility(0.25);
	this->SetRiskFreeRate(0.1);
}

//Constructor
__device__ __host__ Input_market_data::Input_market_data(float ZeroPrice, float Volatility, float RiskFreeRate){
	this->SetZeroPrice(ZeroPrice);
	this->SetVolatility(Volatility);
	this->SetRiskFreeRate(RiskFreeRate);
}

//Copy constructor
__device__ __host__ Input_market_data::Input_market_data(const Input_market_data& data){
	this->SetZeroPrice(data.GetZeroPrice());
	this->SetVolatility(data.GetVolatility());
	this->SetRiskFreeRate(data.GetRiskFreeRate());

}
//Methods
__device__ __host__ void Input_market_data::SetZeroPrice(float ZeroPrice){
	_ZeroPrice = ZeroPrice;
}

__device__ __host__ float Input_market_data::GetZeroPrice() const{
	return _ZeroPrice;
}

__device__ __host__ void Input_market_data::SetVolatility(float Volatility){
	_Volatility = Volatility;
}

__device__ __host__ float Input_market_data::GetVolatility() const{
	return _Volatility;
}

__device__ __host__ void Input_market_data::SetRiskFreeRate(float RiskFreeRate){
	_RiskFreeRate = RiskFreeRate;
}

__device__ __host__ float Input_market_data::GetRiskFreeRate() const{
	return _RiskFreeRate;
}
