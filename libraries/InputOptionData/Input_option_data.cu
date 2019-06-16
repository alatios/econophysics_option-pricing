#include "Input_option_data.cuh"

using namespace std;

//Default Constructor
__device__ __host__ Input_option_data::Input_option_data(){
	this->SetStrikePrice(110.);
	this->SetTimeToMaturity(365.);
	this->SetNumberOfIntervals(365);
	this->SetOptionType('c');
}

//Constructor
__device__ __host__ Input_option_data::Input_option_data(double StrikePrice, unsigned int NumberOfIntervals, double TimeToMaturity, char OptionType){
	this->SetStrikePrice(StrikePrice);
	this->SetNumberOfIntervals(NumberOfIntervals);
	this->SetTimeToMaturity(TimeToMaturity);
	this->SetOptionType(OptionType);
}

//Copy constructor
__device__ __host__ Input_option_data::Input_option_data(const Input_option_data& option){
	this->SetStrikePrice(option.GetStrikePrice());
	this->SetNumberOfIntervals(option.GetNumberOfIntervals());
	this->SetTimeToMaturity(option.GetTimeToMaturity());
	this->SetOptionType(option.GetOptionType());
}

//Methods
__device__ __host__ void Input_option_data::SetStrikePrice(double StrikePrice){
	_StrikePrice = StrikePrice;
}

__device__ __host__ double Input_option_data::GetStrikePrice() const{
	return _StrikePrice;
}

__device__ __host__ void Input_option_data::SetNumberOfIntervals(unsigned int NumberOfIntervals){
	_NumberOfIntervals = NumberOfIntervals;
}

__device__ __host__ unsigned int Input_option_data::GetNumberOfIntervals() const{
	return _NumberOfIntervals;
}

__device__ __host__ void Input_option_data::SetTimeToMaturity(double TimeToMaturity){
		_TimeToMaturity = TimeToMaturity;
}

__device__ __host__ double Input_option_data::GetTimeToMaturity() const{
	return _TimeToMaturity;
}

__device__ __host__ void Input_option_data::SetOptionType(const char OptionType){
	_OptionType = OptionType;
}
__device__ __host__ char Input_option_data::GetOptionType() const{
	return _OptionType;
}
