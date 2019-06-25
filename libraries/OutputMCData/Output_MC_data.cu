#include "Output_MC_data.cuh"

using namespace std;

//Constructors
__device__ __host__ Output_MC_data::Output_MC_data(){
	this->SetEstimatedPriceMC(1.);
	this->SetErrorMC(1.);
	this->SetTick(1.);
}

__device__ __host__ Output_MC_data::Output_MC_data(double EstimatedPriceMC, double ErrorMC, double Tock){
	this->SetEstimatedPriceMC(EstimatedPriceMC);
	this->SetErrorMC(ErrorMC);
	this->SetTick(Tock);
}

// Standard get/set methods
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

__device__ __host__ void Output_MC_data::SetTick(double Tock){
	_Tick = Tock;
}

__device__ __host__ double Output_MC_data::GetTick() const{
	return _Tick;
}

// Private methods for Black-Scholes set
__device__ __host__ void Output_MC_data::SetBlackScholesPrice(double BlackScholesPrice){
	_BlackScholesPrice = BlackScholesPrice;
}

__device__ __host__ void Output_MC_data::SetErrorBlackScholes(double errorBlackScholes){
	_ErrorBlackScholes = errorBlackScholes;
}

// Public methods for Black-Scholes get
__device__ __host__ double Output_MC_data::GetBlackScholesPrice(){
	return _BlackScholesPrice;
}

__device__ __host__ double Output_MC_data::GetErrorBlackScholes(){
	return _ErrorBlackScholes;
}

// Private methods for Black-Scholes evaluation
__device__ __host__ void Output_MC_data::BlackScholesCallOption(const Input_option_data& option, const Input_market_data& market){

	double tmp1 = (1./ (market.GetVolatility() * sqrt(option.GetTimeToMaturity()))) * (log(market.GetInitialPrice()/option.GetStrikePrice())
	+ market.GetRiskFreeRate()
	+ (pow(market.GetVolatility(),2)/2.) * option.GetTimeToMaturity());
	
	double tmp2 = tmp1 - market.GetVolatility() * sqrt(option.GetTimeToMaturity());

	double N_tmp1 = 0.5 * (1. + erf(tmp1/sqrt(2.)));
	double N_tmp2 = 0.5 * (1. + erf(tmp2/sqrt(2.)));

	double CallPrice = market.GetInitialPrice() * N_tmp1
	- option.GetStrikePrice() * exp(- market.GetRiskFreeRate()*option.GetTimeToMaturity()) * N_tmp2;

	_BlackScholesPrice = CallPrice;
}

__device__ __host__ void Output_MC_data::BlackScholesPutOption(const Input_option_data& option, const Input_market_data& market){

	double tmp1 = (1./ (market.GetVolatility() * sqrt(option.GetTimeToMaturity()))) * (log(market.GetInitialPrice()/option.GetStrikePrice())
	+ market.GetRiskFreeRate()
	+ (pow(market.GetVolatility(),2)/2.) * option.GetTimeToMaturity());
	
	double tmp2 = tmp1 - market.GetVolatility() * sqrt(option.GetTimeToMaturity());

	double N_tmp1 = 0.5 * (1. + erf(tmp1/sqrt(2.)));
	double N_tmp2 = 0.5 * (1. + erf(tmp2/sqrt(2.)));

	double PutPrice = market.GetInitialPrice() * (N_tmp1 -1.)
	- option.GetStrikePrice() * exp(- market.GetRiskFreeRate()*option.GetTimeToMaturity()) * (N_tmp2 - 1);

	_BlackScholesPrice = PutPrice;
}

__device__ __host__ void Output_MC_data::EvaluateBlackScholesPrice(const Input_option_data& option, const Input_market_data& market){
	if(option.GetOptionType() == 'c')
		this->BlackScholesCallOption(option, market);
	else if(option.GetOptionType() == 'p')
		this->BlackScholesPutOption(option, market);
	else
		_BlackScholesPrice = 0.;	
}

__device__ __host__ void Output_MC_data::EvaluateErrorBlackScholes(){
	double absoluteValue = fabs(this->GetEstimatedPriceMC() - this->GetBlackScholesPrice());
	this->SetErrorBlackScholes(absoluteValue / this->GetErrorMC());
}

// Public methods for Black-Scholes evaluation
__device__ __host__ void Output_MC_data::CompleteEvaluationOfBlackScholes(const Input_option_data& option, const Input_market_data& market){
	this->EvaluateBlackScholesPrice(option, market);
	this->EvaluateErrorBlackScholes();
}

// Public methods for output management
__host__ void Output_MC_data::PrintResults(){
	cout << "###### OUTPUT DATA ######" << endl << endl;
	cout << "MC estimated price [USD] = " << this->GetEstimatedPriceMC() << endl;
	cout << "MC error [USD] = " << this->GetErrorMC() << endl;
	cout << "Elapsed time [ms] = " << this->GetTick() << endl;
	cout << "Black-Scholes estimated price [USD] = " << this->GetBlackScholesPrice() << endl;
	cout << "Black-Scholes to MC discrepancy [unit of MC sigmas] = " << this->GetErrorBlackScholes() << endl;
}
