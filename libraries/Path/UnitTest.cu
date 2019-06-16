#include "Path.cuh"
#include <iostream>

using namespace std;
/*

OUTPUT:

  No problem ----> 1
  There is a problem ----> 0

*/

int main(){

	double SpotPrice = 1.;

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
	Path path2(SpotPrice);
	Path path3(path1);

	cout << endl << "-------------Path_test-------------" << endl;
	cout << "Constructors testing" << endl;
	//Constructor testing
	cout << (bool)(path1.GetGaussianRandomVariable()==static_cast<double>(0.))
	<< "\t" << (bool)(path1.GetSpotPrice()==static_cast<double>(0.)) << endl;

	cout << (bool)(path2.GetGaussianRandomVariable()==static_cast<double>(0.))
	<< "\t" << (bool)(path2.GetSpotPrice()==static_cast<double>(1.)) << endl;

	cout << (bool)(path3.GetGaussianRandomVariable()==static_cast<double>(0.))
	<< "\t" << (bool)(path3.GetSpotPrice()==static_cast<double>(0.)) << endl;
	
	cout << endl;

	//Methods testing
	cout << "Methods testing" << endl;
	Input_market_data other_market(200., 3., 0.6);
	Input_option_data other_option(190., 100, 300., 'p');	// DeltaTime = 3.

	path1.SetGaussianRandomVariable(0.3);
	path1.SetSpotPrice(240.2);

	cout << (bool)(path1.GetGaussianRandomVariable()==static_cast<double>(0.3))
	<< "\t" << (bool)(path1.GetSpotPrice()==static_cast<double>(240.2)) << endl;

	cout << "Eulero Testing" << endl;
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

	Path pathEulero(SpotPriceEulero);
	pathEulero.EuleroStep(marketEulero, optionEulero);
	cout << (bool)(pathEulero.GetSpotPrice()==static_cast<double>(1.1)) << endl;

	return 0;

}
