#include "Output_MC_data.cuh"
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
	Output_MC_data MCData_2(MarketData, OptionData, 1.1, 1.2, 1.4);

	cout << endl << "-------------Output_MC_data_test-------------" << endl;
	cout << "Constructor testing" << endl;
	test = MCData_1.GetEstimatedPriceMC()==static_cast<float>(1.);
	cout << test << endl;
	test = MCData_1.GetErrorMC()==static_cast<float>(1.);
	cout << test << endl;
	test = MCData_1.GetTick()==static_cast<float>(1.);
	cout << test << endl;
	test = MCData_1.GetInputMarketData().GetZeroPrice()==static_cast<float>(100.);
	cout << test << endl;
	test = MCData_1.GetInputMarketData().GetVolatility()==static_cast<float>(0.25);
	cout << test << endl;
	test = MCData_1.GetInputMarketData().GetRiskFreeRate()==static_cast<float>(0.1);
	cout << test << endl;
	test = MCData_1.GetInputOptionData().GetStrikePrice()==static_cast<float>(110.);
	cout << test << endl;
	test = MCData_1.GetInputOptionData().GetTimeToMaturity()==static_cast<float>(365.);
	cout << test << endl;
	test = MCData_1.GetInputOptionData().GetNumberOfIntervals()==static_cast<unsigned int>(365);
	cout << test << endl;
	test = MCData_1.GetInputOptionData().GetOptionType()==static_cast<char>('c');
	cout << test << endl << endl;

	test = MCData_2.GetEstimatedPriceMC()==static_cast<float>(1.1);
	cout << test << endl;
	test = MCData_2.GetErrorMC()==static_cast<float>(1.2);
	cout << test << endl;
	test = MCData_2.GetTick()==static_cast<float>(1.4);
	cout << test << endl;
	test = MCData_2.GetInputMarketData().GetZeroPrice()==static_cast<float>(0.1);
	cout << test << endl;
	test = MCData_2.GetInputMarketData().GetVolatility()==static_cast<float>(0.2);
	cout << test << endl;
	test = MCData_2.GetInputMarketData().GetRiskFreeRate()==static_cast<float>(0.3);
	cout << test << endl;
	test = MCData_2.GetInputOptionData().GetStrikePrice()==static_cast<float>(0.5);
	cout << test << endl;
	test = MCData_2.GetInputOptionData().GetTimeToMaturity()==static_cast<float>(0.6);
	cout << test << endl;
	test = MCData_2.GetInputOptionData().GetNumberOfIntervals()==static_cast<unsigned int>(120);
	cout << test << endl;
	test = MCData_2.GetInputOptionData().GetOptionType()==static_cast<char>('p');
	cout << test << endl << endl;
	
	

	cout << "Methods testing" << endl;
	//Variables for testing BlackScholes formulae
	//Option
	float StrikePrice = 100.;
	unsigned int NumberOfIntervals = 1;
	float TimeToMaturity = 1.;
	//Market
	float ZeroPrice = 100.;
	float Volatility = 0.25;
	float RiskFreeRate = 0.1;

	Input_market_data market(ZeroPrice, Volatility, RiskFreeRate);
	Input_option_data option_c(StrikePrice, NumberOfIntervals, TimeToMaturity, 'c');
	Input_option_data option_p(StrikePrice, NumberOfIntervals, TimeToMaturity, 'p');
	Output_MC_data MCData_c(market, option_c, 10, 2, 1.4);	
	Output_MC_data MCData_p(market, option_p, 5, 0.5, 1.4);	
	

	test = (abs(MCData_c.GetBlackScholesPrice() - static_cast<float>(14.2558)) < 0.0001);
	cout << test << endl;
	test = (abs(MCData_c.GetErrorBlackScholes() - static_cast<float>(2.1279)) < 0.001);
	cout << test << endl;
	
	test = (abs(MCData_p.GetBlackScholesPrice() - static_cast<float>(4.73955)) < 0.0001);
	cout << test << endl;
	test = (abs(MCData_p.GetErrorBlackScholes() - static_cast<float>(0.5209)) < 0.001);
	cout << test << endl;

	return 0;
}
