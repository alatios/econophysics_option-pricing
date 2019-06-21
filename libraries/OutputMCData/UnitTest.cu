#include "Output_MC_data.cuh"
#include "../InputMarketData/Input_market_data.cuh"
#include "../InputOptionData/Input_option_data.cuh"

#include <iostream>

using namespace std;
/*

OUTPUT:

  No problem ----> 1
  There is a problem ----> 0

*/

int main(){
	bool test;
	Output_MC_data MCData_1;

	Output_MC_data MCData_2(1.1, 1.2, 1.4);

	cout << "\n-------------Output_MC_data_test-------------\n";
	cout << "Constructor testing\n";
	test = MCData_1.GetEstimatedPriceMC()==static_cast<double>(1.)
	&& MCData_1.GetErrorMC()==static_cast<double>(1.)
	&& MCData_1.GetTick()==static_cast<double>(1.);
	cout << test << "\t";

	test = MCData_2.GetEstimatedPriceMC()==static_cast<double>(1.1)
	&& MCData_2.GetErrorMC()==static_cast<double>(1.2)
	&& MCData_2.GetTick()==static_cast<double>(1.4);
	cout << test << "\n";

	cout << "\nMethods testing\n";

	//Variables for testing BlackScholes formulae
	//Option
	double StrikePrice = 100.;
	unsigned int NumberOfIntervals = 1;
	double TimeToMaturity = 1.;
	//Market
	double InitialPrice = 100.;
	double Volatility = 0.25;
	double RiskFreeRate = 0.1;

	Input_market_data market(InitialPrice, Volatility, RiskFreeRate);
	Input_option_data option_c(StrikePrice, NumberOfIntervals, TimeToMaturity, 'c');
	Input_option_data option_p(StrikePrice, NumberOfIntervals, TimeToMaturity, 'p');
	Output_MC_data MCData_c(10, 2, 1.4);	
	Output_MC_data MCData_p(5, 0.5, 1.4);	
	
	MCData_c.CompleteEvaluationOfBlackScholes(option_c, market);
	MCData_p.CompleteEvaluationOfBlackScholes(option_p, market);

	test = (abs(MCData_c.GetBlackScholesPrice() - static_cast<double>(14.9758)) < 0.0001);
	cout << test << "\t";
	test = (abs(MCData_c.GetErrorBlackScholes() - static_cast<double>(2.4879)) < 0.001);
	cout << test << "\t";
		
	test = (abs(MCData_p.GetBlackScholesPrice() - static_cast<double>(5.45953)) < 0.00001);
	cout << test << "\t";
	test = (abs(MCData_p.GetErrorBlackScholes() - static_cast<double>(0.919065)) < 0.00001);
	cout << test << "\n";

	return 0;
}
