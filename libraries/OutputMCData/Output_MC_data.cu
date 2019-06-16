#include "Output_MC_data.cuh"

using namespace std;

//Constructor
__device__ __host__ Output_MC_data::Output_MC_data(){
	this->SetEstimatedPriceMC(1.);
	this->SetErrorMC(1.);
	this->SetTick(1.);
	Input_option_data option;
	Input_market_data market;
	this->SetBlackScholesPrice(option, market);
	this->EvaluateErrorBlackScholes(option, market);
}

__device__ __host__ Output_MC_data::Output_MC_data(const Input_option_data& option, const Input_market_data& market, double EstimatedPriceMC, double ErrorMC, double Tock){
	this->SetEstimatedPriceMC(EstimatedPriceMC);
	this->SetErrorMC(ErrorMC);
	this->SetTick(Tock);
	this->SetBlackScholesPrice(option, market);
	this->EvaluateErrorBlackScholes(option, market);
}

//Methods
__device__ __host__ void Output_MC_data::SetEstimatedPriceMC(double EstimatedPriceMC){
	_EstimatedPriceMC = EstimatedPriceMC;
}

__device__ __host__ double Output_MC_data::GetEstimatedPriceMC() const{
	return _EstimatedPriceMC;
}

__device__ __host__ void Output_MC_data::SetErrorMC(double ErrorMC){
	_ErrorMC = ErrorMC;
}

__device__ __host__ double Output_MC_data::GetErrorMC() const{
	return _ErrorMC;
}

__device__ __host__ void Output_MC_data::SetErrorBlackScholes(const Input_option_data& option, const Input_market_data& market){
	_ErrorBlackScholes = abs(this->GetEstimatedPriceMC() - this->GetBlackScholesPrice(option, market)) / this->GetErrorMC();
}

__device__ __host__ void Output_MC_data::EvaluateErrorBlackScholes(const Input_option_data& option, const Input_market_data& market){
	this->SetErrorBlackScholes(option, market);
}

__device__ __host__ double Output_MC_data::GetErrorBlackScholes(){
	return _ErrorBlackScholes;
}

__device__ __host__ void Output_MC_data::SetTick(double Tock){
	_Tick = Tock;
}

__device__ __host__ double Output_MC_data::GetTick() const{
	return _Tick;
}

__device__ __host__ double Output_MC_data::GetBlackScholesPrice(const Input_option_data& option, const Input_market_data& market){
	this->SetBlackScholesPrice(option, market);
	return _BlackScholesPrice;
}

__device__ __host__ void Output_MC_data::SetBlackScholesPrice(const Input_option_data& option, const Input_market_data& market){
	if(option.GetOptionType() == 'c')
		this->BlackScholesCallOption(option, market);
	else if(option.GetOptionType() == 'p')
		this->BlackScholesPutOption(option, market);
	else
		_BlackScholesPrice = 0.;
}

__device__ __host__ void Output_MC_data::BlackScholesCallOption(const Input_option_data& option, const Input_market_data& market){

	double tmp1 = (1./ (market.GetVolatility() * sqrt(static_cast<double>(static_cast<double>(option.GetTimeToMaturity())
	/ static_cast<double>(option.GetNumberOfIntervals()))))) * (log(market.GetInitialPrice()/option.GetStrikePrice())
	+ market.GetRiskFreeRate()
	+ (pow(market.GetVolatility(),2)/2.) * option.GetTimeToMaturity());
	
	double tmp2 = tmp1 - market.GetVolatility() * sqrt(option.GetTimeToMaturity());

	double N_tmp1 = 0.5 * (1. + erf(tmp1/sqrt(2.)));
	double N_tmp2 = 0.5 * (1. + erf(tmp2/sqrt(2.)));

	double CallPrice = market.GetInitialPrice() * N_tmp1
	- option.GetStrikePrice() * exp(- market.GetRiskFreeRate()*option.GetTimeToMaturity()) * N_tmp2;

	_BlackScholesPrice = CallPrice;
}

//static_cast<double>(static_cast<double>(option.GetTimeToMaturity()) / static_cast<double>(option.GetNumberOfIntervals())))

__device__ __host__ void Output_MC_data::BlackScholesPutOption(const Input_option_data& option, const Input_market_data& market){

	double tmp1 = (1./ (market.GetVolatility() * sqrt(static_cast<double>(static_cast<double>(option.GetTimeToMaturity())
		/ static_cast<double>(option.GetNumberOfIntervals()))))) * (log(market.GetInitialPrice()/option.GetStrikePrice())
	+ market.GetRiskFreeRate()
	+ (pow(market.GetVolatility(),2)/2.) * option.GetTimeToMaturity());
	
	double tmp2 = tmp1 - market.GetVolatility() * sqrt(option.GetTimeToMaturity());

	double N_tmp1 = 0.5 * (1. + erf(tmp1/sqrt(2.)));
	double N_tmp2 = 0.5 * (1. + erf(tmp2/sqrt(2.)));

	double PutPrice = market.GetInitialPrice() * (N_tmp1 -1.)
	- option.GetStrikePrice() * exp(- market.GetRiskFreeRate()*option.GetTimeToMaturity()) * (N_tmp2 - 1);

	_BlackScholesPrice = PutPrice;
}