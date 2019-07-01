#include "Path.cuh"
#include <iostream>

using namespace std;
/*

OUTPUT:

  No problem ----> 1
  There is a problem ----> 0

*/

int main(){

/*	double SpotPrice = 1.;
	bool test;

	//Variables for testing Eulero formulae
	//Market
	double InitialPrice = 150.;
	double Volatility = 0.35;
	double RiskFreeRate = 0.27;
	//Option
	double StrikePrice = 200.;
	unsigned int NumberOfIntervals = 4;
	double TimeToMaturity = 200.;	// DeltaTime = 50
	Input_market_data market(InitialPrice, Volatility, RiskFreeRate);
	Input_option_data option(StrikePrice, NumberOfIntervals, TimeToMaturity, 'p');

	Path path1;
	Path path2(market, option, SpotPrice);
	Path path3(path1);

	cout << "\n-------------Path_test-------------\n";
	cout << "Constructors testing\n";
	//Constructor testing
	test = path1.GetSpotPrice() == static_cast<double>(0.);
	cout << test << "\t";
	test = path2.GetSpotPrice() == static_cast<double>(1.);
	cout << test << "\t";
	test = path3.GetSpotPrice() == static_cast<double>(0.);
	cout << test << "\n";

	//Methods testing
	cout << "\nMethods testing\n";
	cout << "Eulero Testing\n";
	//Variables for testing Eulero formulae
	//Option
	double SpotPriceEulero = 1.;
	double StrikePriceEulero = 100.;
	unsigned int NumberOfIntervalsEulero = 1;
	double TimeToMaturityEulero = 1.;
	//Market
	double InitialPriceEulero = 100.;
	double VolatilityEulero = 0.25;
	double RiskFreeRateEulero = 0.1;
	Input_market_data marketEulero(InitialPriceEulero, VolatilityEulero, RiskFreeRateEulero);
	Input_option_data optionEulero(StrikePriceEulero, NumberOfIntervalsEulero, TimeToMaturityEulero, 'p');

	path1.SetInternalState(marketEulero, optionEulero, SpotPriceEulero);
	path1.EuleroStep(0.3);
	
	path2.SetInternalState(path1);

	test = path1.GetSpotPrice() == static_cast<double>(1.175);
	cout << test << "\t";
	test = path2.GetSpotPrice() == static_cast<double>(1.175);
	cout << test << "\n";
*/
	return 0;

}
