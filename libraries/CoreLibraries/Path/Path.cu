#include "Path.cuh"
#include <cmath>

using namespace std;

// Constructors
__device__ __host__ Path::Path(){
	this->_OptionType = NULL;
	this->_SpotPrice = 0.;
	this->_RiskFreeRate = NULL;
	this->_Volatility = NULL;
	this->_TimeToMaturity = NULL;
	this->_NumberOfIntervals = NULL;
	this->_DeltaTime = 0.;
	this->_StrikePrice = NULL;
	this->_B = NULL;
	this->_N = NULL;
	this->_K = NULL;
	this->_PerformanceCorridorBarrierCounter = 0;
}

__device__ __host__ Path::Path(const Input_market_data& market, const Input_option_data& option){
	this->_OptionType = &(option.OptionType);
	this->_SpotPrice = market.InitialPrice;
	this->_RiskFreeRate = &(market.RiskFreeRate);
	this->_Volatility = &(market.Volatility);
	this->_TimeToMaturity = &(option.TimeToMaturity);
	this->_NumberOfIntervals = &(option.NumberOfIntervals);
	this->_DeltaTime = option.GetDeltaTime();
	this->_StrikePrice = &(option.StrikePrice);
	this->_B = &(option.B);
	this->_N = &(option.N);
	this->_K = &(option.K);
	this->_PerformanceCorridorBarrierCounter = 0;
}

// Public set methods
__device__ __host__ void Path::ResetToInitialState(const Input_market_data& market, const Input_option_data& option){
	this->_OptionType = &(option.OptionType);
	this->_SpotPrice = market.InitialPrice;
	this->_RiskFreeRate = &(market.RiskFreeRate);
	this->_Volatility = &(market.Volatility);
	this->_TimeToMaturity = &(option.TimeToMaturity);
	this->_NumberOfIntervals = &(option.NumberOfIntervals);
	this->_DeltaTime = option.GetDeltaTime();
	this->_StrikePrice = &(option.StrikePrice);
	this->_B = &(option.B);
	this->_N = &(option.N);
	this->_K = &(option.K);
	this->_PerformanceCorridorBarrierCounter = 0;
}

__device__ __host__ void Path::ResetToInitialState(const Path& otherPath){
	this->_OptionType = otherPath._OptionType;
	this->_SpotPrice = otherPath._SpotPrice;
	this->_RiskFreeRate = otherPath._RiskFreeRate;
	this->_Volatility = otherPath._Volatility;
	this->_TimeToMaturity = otherPath._TimeToMaturity;
	this->_NumberOfIntervals = otherPath._NumberOfIntervals;
	this->_DeltaTime = otherPath._DeltaTime;
	this->_StrikePrice = otherPath._StrikePrice;
	this->_B = otherPath._B;
	this->_N = otherPath._N;
	this->_K = otherPath._K;
	this->_PerformanceCorridorBarrierCounter = otherPath._PerformanceCorridorBarrierCounter;
}

// Public get methods
__device__ __host__ double Path::GetSpotPrice() const{
	return this->_SpotPrice;
}

__device__ __host__ unsigned int Path::GetPerformanceCorridorBarrierCounter() const{
	return this->_PerformanceCorridorBarrierCounter;
}


// Euler and exact steps implementation
__device__ __host__ void Path::EulerLogNormalStep(double gaussianRandomVariable){
	double SpotPrice_i;		//The price at the next step
	SpotPrice_i = (this->_SpotPrice) *
	(1 + *(this->_RiskFreeRate) * this->_DeltaTime
	+ *(this->_Volatility) * sqrt(this->_DeltaTime) * gaussianRandomVariable);
	
	if(*(_OptionType) == 'e')
		this->CheckPerformanceCorridorCondition(this->_SpotPrice, SpotPrice_i);
	
	this->_SpotPrice = SpotPrice_i;
}

__device__ __host__ void Path::ExactLogNormalStep(double gaussianRandomVariable){
	double SpotPrice_i;		//The price at the next step
	SpotPrice_i = (this->_SpotPrice) * exp((*(this->_RiskFreeRate)
	- 0.5 * pow(*(this->_Volatility),2)) * this->_DeltaTime
	+ *(this->_Volatility) * gaussianRandomVariable * sqrt(this->_DeltaTime));
	
	if(*(_OptionType) == 'e')
		this->CheckPerformanceCorridorCondition(this->_SpotPrice, SpotPrice_i);
	
	this->_SpotPrice = SpotPrice_i;
}

// Check performance corridor condition
__device__ __host__ void Path::CheckPerformanceCorridorCondition(double currentSpotPrice, double nextSpotPrice){
	double modulusArgument = 1./(sqrt(this->_DeltaTime)) * log(nextSpotPrice / currentSpotPrice);
	double barrier = *(this->_B) * *(this->_Volatility);

	if(fabs(modulusArgument) < barrier)
		++(this->_PerformanceCorridorBarrierCounter);
}

// Evaluate atualized payoff
__device__ __host__ double Path::GetActualizedPayoff() const{
	double payoff;
	
	switch(*(this->_OptionType)){
		case 'f':
			payoff = this->_SpotPrice;
			break;
		
		case 'c':
			payoff = fmax(this->_SpotPrice - *(this->_StrikePrice), 0.);
			break;
		
		case 'p':
			payoff = fmax(*(this->_StrikePrice) - this->_SpotPrice, 0.);
			break;
		
		case 'e':
			payoff = *(this->_N) * fmax((static_cast<double>(this->_PerformanceCorridorBarrierCounter) / *(this->_NumberOfIntervals)) - *(this->_K), 0.);
			break;
			
		default:
			payoff = -10000.;
			break;
	}	
	
	return (payoff * exp(- *(this->_RiskFreeRate) * *(this->_TimeToMaturity)));
}
