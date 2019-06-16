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
	
	Input_market_data MarketData(0.1,0.2,0.3);
	Input_option_data OptionData(0.5, 120, 0.6, 'p');
	Output_MC_data MCData_2(OptionData, MarketData, 1.1, 1.2, 1.4);

	cout << endl << "-------------Output_MC_data_test-------------" << endl;
	cout << "Constructor testing" << endl;
	test = MCData_1.GetEstimatedPriceMC()==static_cast<double>(1.);
	cout << test << endl;
	test = MCData_1.GetErrorMC()==static_cast<double>(1.);
	cout << test << endl;
	test = MCData_1.GetTick()==static_cast<double>(1.);
	cout << test << endl;

	test = MCData_2.GetEstimatedPriceMC()==static_cast<double>(1.1);
	cout << test << endl;
	test = MCData_2.GetErrorMC()==static_cast<double>(1.2);
	cout << test << endl;
	test = MCData_2.GetTick()==static_cast<double>(1.4);
	cout << test << endl;

	cout << "Methods testing" << endl;
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
	Output_MC_data MCData_c(option_c, market, 10, 2, 1.4);	
	Output_MC_data MCData_p(option_p, market, 5, 0.5, 1.4);	
	

	test = (abs(MCData_c.GetBlackScholesPrice(option_c, market) - static_cast<double>(14.9758)) < 0.0001);
	cout << test << endl;
	MCData_c.EvaluateErrorBlackScholes(option_c, market);
	test = (abs(MCData_c.GetErrorBlackScholes() - static_cast<double>(2.4878)) < 0.001);
	cout << test << endl;
	
	test = (abs(MCData_p.GetBlackScholesPrice(option_p, market) - static_cast<double>(5.45953)) < 0.00001);
	cout << test << endl;
	MCData_p.EvaluateErrorBlackScholes(option_p, market);
	test = (abs(MCData_p.GetErrorBlackScholes() - static_cast<double>(0.919065)) < 0.00001);
	cout << test << endl;

/*
	test = (abs(MCData_c.GetBlackScholesPrice(option_c, market) - static_cast<double>(14.2558)) < 0.0001);
	cout << test << endl;
	MCData_c.EvaluateErrorBlackScholes(option_c, market);
	test = (abs(MCData_c.GetErrorBlackScholes() - static_cast<double>(2.1279)) < 0.001);
	cout << test << endl;
	
	test = (abs(MCData_p.GetBlackScholesPrice(option_p, market) - static_cast<double>(4.73955)) < 0.0001);
	cout << test << endl;
	MCData_p.EvaluateErrorBlackScholes(option_p, market);
	test = (abs(MCData_p.GetErrorBlackScholes() - static_cast<double>(0.5209)) < 0.001);
	cout << test << endl;
*/
	return 0;
}
