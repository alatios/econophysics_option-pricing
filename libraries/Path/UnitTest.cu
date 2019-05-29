#include "Path.cuh"
#include <iostream>

using namespace std;
/*

OUTPUT:

  No problem ----> 1
  There is a problem ----> 0

*/

int main(){

	float SpotPrice = 1.;

	//Variables for testing Eulero formulae
	//Market
	float ZeroPrice = 150.;
	float Volatility = 0.35;
	float RiskFreeRate = 0.27;
	//Option
	float StrikePrice = 200.;
	unsigned int NumberOfIntervals = 4;
	float TimeToMaturity = 200.;	// DeltaTime = 50
	Input_market_data market(ZeroPrice, Volatility, RiskFreeRate);
	Input_option_data option(StrikePrice, NumberOfIntervals, TimeToMaturity, 'p');

	Path path1;
	Path path2(market, option, SpotPrice);
	Path path3(path1);

	cout << endl << "-------------Path_test-------------" << endl;
	cout << "Constructors testing" << endl;
	//Constructor testing
	cout << (bool)(path1.GetGaussianRandomVariable()==static_cast<float>(0.))
	<< "\t" << (bool)(path1.GetSpotPrice()==static_cast<float>(0.))
	<< "\t" << (bool)(path1.GetInputMarketData().GetZeroPrice()==static_cast<float>(100.))
	<< "\t" << (bool)(path1.GetInputMarketData().GetVolatility()==static_cast<float>(0.25))
	<< "\t" << (bool)(path1.GetInputMarketData().GetRiskFreeRate()==static_cast<float>(0.1))
	<< "\t" << (bool)(path1.GetInputOptionData().GetStrikePrice()==static_cast<float>(110.))
	<< "\t" << (bool)(path1.GetInputOptionData().GetNumberOfIntervals()==static_cast<unsigned int>(365))
	<< "\t" << (bool)(path1.GetInputOptionData().GetTimeToMaturity()==static_cast<float>(365.))
	<< "\t" << (bool)(path1.GetInputOptionData().GetDeltaTime()==static_cast<float>(1.))
	<< "\t" << (bool)(path1.GetInputOptionData().GetOptionType()==static_cast<char>('c')) << endl;

	cout << (bool)(path2.GetGaussianRandomVariable()==static_cast<float>(0.))
	<< "\t" << (bool)(path2.GetSpotPrice()==static_cast<float>(1.))
	<< "\t" << (bool)(path2.GetInputMarketData().GetZeroPrice()==static_cast<float>(150.))
	<< "\t" << (bool)(path2.GetInputMarketData().GetVolatility()==static_cast<float>(0.35))
	<< "\t" << (bool)(path2.GetInputMarketData().GetRiskFreeRate()==static_cast<float>(0.27))
	<< "\t" << (bool)(path2.GetInputOptionData().GetStrikePrice()==static_cast<float>(200.))
	<< "\t" << (bool)(path2.GetInputOptionData().GetNumberOfIntervals()==static_cast<unsigned int>(4))
	<< "\t" << (bool)(path2.GetInputOptionData().GetTimeToMaturity()==static_cast<float>(200.))
	<< "\t" << (bool)(path2.GetInputOptionData().GetDeltaTime()==static_cast<float>(50.))
	<< "\t" << (bool)(path2.GetInputOptionData().GetOptionType()==static_cast<char>('p')) << endl;

	cout << (bool)(path3.GetGaussianRandomVariable()==static_cast<float>(0.))
	<< "\t" << (bool)(path3.GetSpotPrice()==static_cast<float>(0.))
	<< "\t" << (bool)(path3.GetInputMarketData().GetZeroPrice()==static_cast<float>(100.))
	<< "\t" << (bool)(path3.GetInputMarketData().GetVolatility()==static_cast<float>(0.25))
	<< "\t" << (bool)(path3.GetInputMarketData().GetRiskFreeRate()==static_cast<float>(0.1))
	<< "\t" << (bool)(path3.GetInputOptionData().GetStrikePrice()==static_cast<float>(110.))
	<< "\t" << (bool)(path3.GetInputOptionData().GetNumberOfIntervals()==static_cast<unsigned int>(365))
	<< "\t" << (bool)(path3.GetInputOptionData().GetTimeToMaturity()==static_cast<float>(365.))
	<< "\t" << (bool)(path3.GetInputOptionData().GetDeltaTime()==static_cast<float>(1.))
	<< "\t" << (bool)(path3.GetInputOptionData().GetOptionType()==static_cast<char>('c')) << endl;
	cout << endl;

	//Methods testing
	cout << "Methods testing" << endl;
	Input_market_data other_market(200., 3., 0.6);
	Input_option_data other_option(190., 100, 300., 'p');	// DeltaTime = 3.

	path1.SetGaussianRandomVariable(0.3);
	path1.SetSpotPrice(240.2);
	path1.SetInputMarketData(other_market);
	path1.SetInputOptionData(other_option);

	cout << (bool)(path1.GetGaussianRandomVariable()==static_cast<float>(0.3))
	<< "\t" << (bool)(path1.GetSpotPrice()==static_cast<float>(240.2))
	<< "\t" << (bool)(path1.GetInputMarketData().GetZeroPrice()==static_cast<float>(200.))
	<< "\t" << (bool)(path1.GetInputMarketData().GetVolatility()==static_cast<float>(3.))
	<< "\t" << (bool)(path1.GetInputMarketData().GetRiskFreeRate()==static_cast<float>(0.6))
	<< "\t" << (bool)(path1.GetInputOptionData().GetStrikePrice()==static_cast<float>(190.))
	<< "\t" << (bool)(path1.GetInputOptionData().GetNumberOfIntervals()==static_cast<unsigned int>(100))
	<< "\t" << (bool)(path1.GetInputOptionData().GetTimeToMaturity()==static_cast<float>(300.))
	<< "\t" << (bool)(path1.GetInputOptionData().GetDeltaTime()==static_cast<float>(3.))
	<< "\t" << (bool)(path1.GetInputOptionData().GetOptionType()==static_cast<char>('p')) << endl;

	cout << "Eulero Testing" << endl;
	//Variables for testing Eulero formulae
	//Option
	float SpotPriceEulero = 1.;
	float StrikePriceEulero = 100.;
	unsigned int NumberOfIntervalsEulero = 1;
	float TimeToMaturityEulero = 1.;
	//Market
	float ZeroPriceEulero = 100.;
	float VolatilityEulero = 0.25;
	float RiskFreeRateEulero = 0.1;
	Input_market_data marketEulero(ZeroPriceEulero, VolatilityEulero, RiskFreeRateEulero);
	Input_option_data optionEulero(StrikePriceEulero, NumberOfIntervalsEulero, TimeToMaturityEulero, 'p');

	Path pathEulero(marketEulero, optionEulero, SpotPriceEulero);
	pathEulero.EuleroStep();
	cout << (bool)(pathEulero.GetSpotPrice()==static_cast<float>(1.1)) << endl;

	return 0;

}
