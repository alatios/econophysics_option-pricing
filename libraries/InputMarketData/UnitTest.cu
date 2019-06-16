#include "Input_market_data.cuh"
#include <iostream>
#include <iomanip>

using namespace std;
/*

OUTPUT:

  No problem ----> 1
  There is a problem ----> 0

*/

int main(){

	//Random choice only for testing
	double InitialPrice = 0.;
	double Volatility = 1.;
	double RiskFreeRate = 2.;
	bool test;

//	Input_market_data MarketData_1 = Input_market_data();
	Input_market_data MarketData_1;
	Input_market_data MarketData_2 = Input_market_data(InitialPrice, Volatility, RiskFreeRate);
	Input_market_data MarketData_3(MarketData_1);

	cout << endl << "-------------Input_market_data_test-------------" << endl;
	cout << "Constructors testing" << endl;
	//Constructor testing
	test = (MarketData_1.GetInitialPrice()==static_cast<double>(100.) && MarketData_1.GetVolatility()==static_cast<double>(0.25) && MarketData_1.GetRiskFreeRate()==static_cast<double>(0.1));
	cout << test << endl;

	test = (MarketData_2.GetInitialPrice()==static_cast<double>(0.) && MarketData_2.GetVolatility()==static_cast<double>(1.) && MarketData_2.GetRiskFreeRate()==static_cast<double>(2.));
	cout << test << endl;

	test = (MarketData_3.GetInitialPrice()==static_cast<double>(100.) && MarketData_3.GetVolatility()==static_cast<double>(0.25) && MarketData_3.GetRiskFreeRate()==static_cast<double>(0.1));
	cout << test << endl;

	cout << "Methods testing" << endl;
	//Methods testing
	MarketData_1.SetInitialPrice(3.);
	MarketData_1.SetVolatility(4.);
	MarketData_1.SetRiskFreeRate(5.);

	test = MarketData_1.GetInitialPrice()==static_cast<double>(3.);
	cout << test << endl;
	test = MarketData_1.GetVolatility()==static_cast<double>(4.);
	cout << test << endl;
	test = MarketData_1.GetRiskFreeRate()==static_cast<double>(5.);
	cout << test << endl;

	return 0;
}
