#include "Input_market_data.cuh"

using namespace std;


//Default Constructor
__device__ __host__ Input_market_data::Input_market_data(){
	this->SetInitialPrice(100.);
	this->SetVolatility(0.25);
	this->SetRiskFreeRate(0.1);
}

//Constructor
__device__ __host__ Input_market_data::Input_market_data(double InitialPrice, double Volatility, double RiskFreeRate){
	this->SetInitialPrice(InitialPrice);
	this->SetVolatility(Volatility);
	this->SetRiskFreeRate(RiskFreeRate);
}

//Copy constructor
__device__ __host__ Input_market_data::Input_market_data(const Input_market_data& data){
	this->SetInitialPrice(data.GetInitialPrice());
	this->SetVolatility(data.GetVolatility());
	this->SetRiskFreeRate(data.GetRiskFreeRate());

}
//Methods
__device__ __host__ void Input_market_data::SetInitialPrice(double InitialPrice){
	_InitialPrice = InitialPrice;
}

__device__ __host__ double Input_market_data::GetInitialPrice() const{
	return _InitialPrice;
}

__device__ __host__ void Input_market_data::SetVolatility(double Volatility){
	_Volatility = Volatility;
}

__device__ __host__ double Input_market_data::GetVolatility() const{
	return _Volatility;
}

__device__ __host__ void Input_market_data::SetRiskFreeRate(double RiskFreeRate){
	_RiskFreeRate = RiskFreeRate;
}

__device__ __host__ double Input_market_data::GetRiskFreeRate() const{
	return _RiskFreeRate;
}

__host__ void Input_market_data::PrintMarketInput() const{
	cout << "Initial underlying price [USD]: " << this->GetInitialPrice() << endl;
	cout << "Market volatility: " << this->GetVolatility() << endl;
	cout << "Risk free rate: " << this->GetRiskFreeRate() << endl;	
}
