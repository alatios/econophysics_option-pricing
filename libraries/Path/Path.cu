#include "Path.cuh"

using namespace std;

__device__ __host__ Path::Path(){
	this->SetSpotPrice(0.);
	Input_market_data MarketData;
	Input_option_data OptionData;
	this->SetInputMarketData(MarketData);
	this->SetInputOptionData(OptionData);
	this->SetGaussianRandomVariable(0.);
}

//Constructor
__device__ __host__ Path::Path(const Input_market_data& MarketData, const Input_option_data& OptionData, float SpotPrice){
	this->SetSpotPrice(SpotPrice);
	this->SetInputMarketData(MarketData);
	this->SetInputOptionData(OptionData);
	this->SetGaussianRandomVariable(0.);
}
//Copy Constructor
__device__ __host__ Path::Path(const Path& p){
	this->SetGaussianRandomVariable(p.GetGaussianRandomVariable());
	this->SetSpotPrice(p.GetSpotPrice());
	this->SetInputMarketData(p.GetInputMarketData());
	this->SetInputOptionData(p.GetInputOptionData());
}

//Methods
__device__ __host__ float Path::GetSpotPrice() const{
	return _SpotPrice;
}

__device__ __host__ void Path::SetSpotPrice(float SpotPrice){
	_SpotPrice = SpotPrice;
}

__device__ __host__ float Path::GetGaussianRandomVariable() const{
	return _GaussianRandomVariable;
}

__device__ __host__ void Path::SetGaussianRandomVariable(float GaussianRandomVariable){
	_GaussianRandomVariable = GaussianRandomVariable;
}

__device__ __host__ void Path::SetInputMarketData(const Input_market_data& MarketData){
	_MarketData.SetZeroPrice(MarketData.GetZeroPrice());
	_MarketData.SetVolatility(MarketData.GetVolatility());
	_MarketData.SetRiskFreeRate(MarketData.GetRiskFreeRate());
}

__device__ __host__ Input_market_data Path::GetInputMarketData() const{
	return _MarketData;
}

__device__ __host__ void Path::SetInputOptionData(const Input_option_data& OptionData){
	_OptionData.SetOptionType(OptionData.GetOptionType());
	_OptionData.SetStrikePrice(OptionData.GetStrikePrice());
	_OptionData.SetNumberOfIntervals(OptionData.GetNumberOfIntervals());
	_OptionData.SetTimeToMaturity(OptionData.GetTimeToMaturity());
	_OptionData.SetDeltaTime();
}

__device__ __host__ Input_option_data Path::GetInputOptionData() const{
	return _OptionData;
}

__device__ __host__ void Path::EuleroStep(){			//It takes a step according to the euler formula
	Input_market_data market = this->GetInputMarketData();
	Input_option_data option = this->GetInputOptionData();
	float SpotPrice_i;		//The price at the next step
	SpotPrice_i = (this->GetSpotPrice()) * (1 + market.GetRiskFreeRate() * (option.GetDeltaTime()) + market.GetVolatility() * sqrt(option.GetDeltaTime()) * (this->GetGaussianRandomVariable()));
	this->SetSpotPrice(SpotPrice_i);
}
