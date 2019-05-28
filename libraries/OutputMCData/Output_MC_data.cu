#include "Output_MC_data.cuh"

using namespace std;

//Constructor
__device__ __host__ Output_MC_data::Output_MC_data(){
	this->SetEstimatedPriceMC(1.);
	this->SetErrorMC(1.);
	this->SetTick(1.);
	Input_market_data MarketData;
	Input_option_data OptionData;
	this->SetInputMarketData(MarketData);
	this->SetInputOptionData(OptionData);
	this->SetBlackScholesPrice();
	this->SetErrorBlackScholes();
}

__device__ __host__ Output_MC_data::Output_MC_data(const Input_market_data& MarketData, const Input_option_data& OptionData, float EstimatedPriceMC, float ErrorMC, float Tock){
	this->SetEstimatedPriceMC(EstimatedPriceMC);
	this->SetErrorMC(ErrorMC);
	this->SetTick(Tock);
	this->SetInputMarketData(MarketData);
	this->SetInputOptionData(OptionData);
	this->SetBlackScholesPrice();
	this->SetErrorBlackScholes();
}


//Methods
__device__ __host__ void Output_MC_data::SetEstimatedPriceMC(float EstimatedPriceMC){
	_EstimatedPriceMC = EstimatedPriceMC;
}

__device__ __host__ float Output_MC_data::GetEstimatedPriceMC() const{
	return _EstimatedPriceMC;
}

__device__ __host__ void Output_MC_data::SetErrorMC(float ErrorMC){
	_ErrorMC = ErrorMC;
}

__device__ __host__ float Output_MC_data::GetErrorMC() const{
	return _ErrorMC;
}

__device__ __host__ void Output_MC_data::SetErrorBlackScholes(){
	_ErrorBlackScholes = fabsf(this->GetEstimatedPriceMC() - this->GetBlackScholesPrice())
		/ this->GetErrorMC();
}

__device__ __host__ float Output_MC_data::GetErrorBlackScholes(){
	this->SetErrorBlackScholes();
	return _ErrorBlackScholes;
}

__device__ __host__ void Output_MC_data::SetTick(float Tock){
	_Tick = Tock;
}

__device__ __host__ float Output_MC_data::GetTick() const{
	return _Tick;
}

__device__ __host__ float Output_MC_data::GetBlackScholesPrice(){
	this->SetBlackScholesPrice();
	return _BlackScholesPrice;
}

__device__ __host__ void Output_MC_data::SetBlackScholesPrice(){
	if(this->GetInputOptionData().GetOptionType() == 'c')
		this->BlackScholesCallOption();
	else if(this->GetInputOptionData().GetOptionType() == 'p')
		this->BlackScholesPutOption();
	else
		_BlackScholesPrice = 0.;
}

__device__ __host__ void Output_MC_data::BlackScholesCallOption(){
	Input_market_data market = GetInputMarketData();
	Input_option_data option = GetInputOptionData();

	float tmp1 = (1./ (market.GetVolatility() * sqrtf(option.GetDeltaTime()))) * logf(market.GetZeroPrice()/option.GetStrikePrice())
	+ (market.GetRiskFreeRate()
	+ (powf(market.GetVolatility(),2)/2.) * option.GetTimeToMaturity());
	
	float tmp2 = tmp1 - market.GetVolatility() * sqrtf(option.GetTimeToMaturity());

	float N_tmp1 = 0.5 * (1. + erff(tmp1/sqrtf(2.)));
	float N_tmp2 = 0.5 * (1. + erff(tmp2/sqrtf(2.)));

	float CallPrice = market.GetZeroPrice() * N_tmp1
	- option.GetStrikePrice() * expf(- market.GetRiskFreeRate()*option.GetTimeToMaturity()) * N_tmp2;

	_BlackScholesPrice = CallPrice;
}

__device__ __host__ void Output_MC_data::BlackScholesPutOption(){
	Input_market_data market = GetInputMarketData();
	Input_option_data option = GetInputOptionData();

	float tmp1 = (1./ (market.GetVolatility() * sqrtf(option.GetDeltaTime()))) * logf(market.GetZeroPrice()/option.GetStrikePrice())
	+ (market.GetRiskFreeRate()
	+ (powf(market.GetVolatility(),2)/2.) * option.GetTimeToMaturity());
	
	float tmp2 = tmp1 - market.GetVolatility() * sqrtf(option.GetTimeToMaturity());

	float N_tmp1 = 0.5 * (1. + erff(tmp1/sqrtf(2.)));
	float N_tmp2 = 0.5 * (1. + erff(tmp2/sqrtf(2.)));

	float PutPrice = market.GetZeroPrice() * (N_tmp1 -1.)
	- option.GetStrikePrice() * expf(- market.GetRiskFreeRate()*option.GetTimeToMaturity()) * (N_tmp2 - 1);

	_BlackScholesPrice = PutPrice;
}

__device__ __host__ Input_market_data Output_MC_data::GetInputMarketData() const{
	return _MarketData;
}

__device__ __host__ void Output_MC_data::SetInputMarketData(const Input_market_data& MarketData){
	_MarketData.SetZeroPrice(MarketData.GetZeroPrice());
	_MarketData.SetVolatility(MarketData.GetVolatility());
	_MarketData.SetRiskFreeRate(MarketData.GetRiskFreeRate());
}

__device__ __host__ Input_option_data Output_MC_data::GetInputOptionData() const{
	return _OptionData;
}

__device__ __host__ void Output_MC_data::SetInputOptionData(const Input_option_data& OptionData){
	_OptionData.SetOptionType(OptionData.GetOptionType());
	_OptionData.SetStrikePrice(OptionData.GetStrikePrice());
	_OptionData.SetNumberOfIntervals(OptionData.GetNumberOfIntervals());
	_OptionData.SetTimeToMaturity(OptionData.GetTimeToMaturity());
	_OptionData.SetDeltaTime();
}
