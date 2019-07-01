
#include "Path.cuh"
#include <iostream>

using namespace std;
/*

OUTPUT:

  No problem ----> 1
  There is a problem ----> 0

*/

int main(){

	bool test;

	//Variables for testing Eulero formulae
	//Market
	double initialPrice = 150.;
	double volatility = 0.35;
	double riskFreeRate = 0.27;
	//Option
	char optionType = 'e';
	unsigned int numberOfIntervals = 4;		// DeltaTime = 50
	double timeToMaturity = 200.;	
	double deltaTime = 50;
	// Plain vanilla option data
	double strikePrice = 200.;
	// Performance corridor data
	double b = 1;
	double k = 2;
	double n = 4;
	unsigned int performanceCorridorBarrierCounter = 0;

	Input_market_data market = {initialPrice, volatility, riskFreeRate};

	Input_option_data option = {optionType, numberOfIntervals, timeToMaturity, strikePrice, b, k, n};

	Path path1;
	Path path2(market, option);

	cout << "\n-------------Path_test-------------\n";
	cout << "Constructors testing\n";

	//Constructor testing
	test = (path1.GetSpotPrice() == static_cast<double>(0.)
		&& path1.GetPerformanceCorridorBarrierCounter() == static_cast<unsigned int>(0));
	cout << test << "\t";
	test = (path2.GetSpotPrice() == static_cast<double>(150.)
		&& path2.GetPerformanceCorridorBarrierCounter() == static_cast<unsigned int>(0));
	cout << test << "\n";

	//Methods testing
	cout << "\nMethods testing\n";

	//Option
	double spotPriceLogNormal = 1.;
	double strikePriceLogNormal = 100.;
	unsigned int numberOfIntervalsLogNormal = 1;
	double timeToMaturityLogNormal = 1.;
	//Market
	double initialPriceLogNormal = 1.;
	double volatilityLogNormal = 0.25;
	double riskFreeRateLogNormal = 0.1;
	Input_market_data marketLogNormal = {initialPriceLogNormal, volatilityLogNormal, riskFreeRateLogNormal};
	Input_option_data optionLogNormal_f = {'f', numberOfIntervalsLogNormal, timeToMaturityLogNormal, 0, 0, 0, 0};
	Input_option_data optionLogNormal_c = {'c', numberOfIntervalsLogNormal, timeToMaturityLogNormal, strikePriceLogNormal, 0, 0, 0};
	Input_option_data optionLogNormal_p = {'p', numberOfIntervalsLogNormal, timeToMaturityLogNormal, strikePriceLogNormal, 0, 0, 0};
	Input_option_data optionLogNormal_e = {'e', numberOfIntervalsLogNormal, timeToMaturityLogNormal, 0, 5, 4, 4.5};

	double gaussianRandomVariable = 0.3;
	
	cout << "ResetToInitialState testing\n";

	path1.ResetToInitialState(marketLogNormal, optionLogNormal_f);
	path2.ResetToInitialState(path1);

	test = (path1.GetSpotPrice() == static_cast<double>(1.)
		&& path1.GetPerformanceCorridorBarrierCounter() == static_cast<unsigned int>(0));
	cout << test << "\t";
	test = (path1.GetSpotPrice() == static_cast<double>(1.)
		&& path1.GetPerformanceCorridorBarrierCounter() == static_cast<unsigned int>(0));
	cout << test << "\n";

	cout << "\nForward option: LogNormalSteps testing\n";

	path1.EulerLogNormalStep(gaussianRandomVariable);
	path2.ExactLogNormalStep(gaussianRandomVariable);

	test = (path1.GetSpotPrice() == static_cast<double>(1.175)
		&& path1.GetPerformanceCorridorBarrierCounter() == static_cast<unsigned int>(0));
	cout << test << "\t";
	test = (path2.GetSpotPrice() - static_cast<double>(1.1546) < static_cast<double>(0.0001)
		&& path2.GetPerformanceCorridorBarrierCounter() == static_cast<unsigned int>(0));
	cout << test << "\n";
	test = (path2.GetActualizedPayoff() - static_cast<double>(1.04472) < static_cast<double>(0.00001));

	cout << "\nPlain vanilla call option: LogNormalSteps testing\n";

	path1.ResetToInitialState(marketLogNormal, optionLogNormal_c);
	path2.ResetToInitialState(path1);
	path1.EulerLogNormalStep(gaussianRandomVariable);
	path2.ExactLogNormalStep(gaussianRandomVariable);

	test = (path1.GetSpotPrice() == static_cast<double>(1.175)
		&& path1.GetPerformanceCorridorBarrierCounter() == static_cast<unsigned int>(0));
	cout << test << "\t";
	test = (path2.GetSpotPrice() - static_cast<double>(1.1546) < static_cast<double>(0.0001)
		&& path2.GetPerformanceCorridorBarrierCounter() == static_cast<unsigned int>(0));
	cout << test << "\t\t";
	test = (path2.GetActualizedPayoff() == static_cast<double>(0.));
	cout << test << "\n";

	cout << "\nPlain vanilla put option: LogNormalSteps testing\n";

	path1.ResetToInitialState(marketLogNormal, optionLogNormal_p);
	path2.ResetToInitialState(path1);
	path1.EulerLogNormalStep(gaussianRandomVariable);
	path2.ExactLogNormalStep(gaussianRandomVariable);

	test = (path1.GetSpotPrice() == static_cast<double>(1.175)
		&& path1.GetPerformanceCorridorBarrierCounter() == static_cast<unsigned int>(0));
	cout << test << "\t";
	test = (path2.GetSpotPrice() - static_cast<double>(1.1546) < static_cast<double>(0.0001)
		&& path2.GetPerformanceCorridorBarrierCounter() == static_cast<unsigned int>(0));
	cout << test << "\t\t";
	test = (path2.GetActualizedPayoff() - static_cast<double>(89.493) < static_cast<double>(0.001));
	cout << test << "\n";

	cout << "\nPerformance corridor option: LogNormalSteps testing\n";

	path1.ResetToInitialState(marketLogNormal, optionLogNormal_e);
	path2.ResetToInitialState(path1);
	path1.EulerLogNormalStep(gaussianRandomVariable);
	path2.ExactLogNormalStep(gaussianRandomVariable);

	test = (path1.GetSpotPrice() == static_cast<double>(1.175)
		&& path1.GetPerformanceCorridorBarrierCounter() == static_cast<unsigned int>(1));
	cout << test << "\t";
	test = (path2.GetSpotPrice() - static_cast<double>(1.1546) < static_cast<double>(0.0001)
		&& path2.GetPerformanceCorridorBarrierCounter() == static_cast<unsigned int>(1));
	cout << test << "\t\t";
	test = (path2.GetActualizedPayoff() == static_cast<double>(0.));
	cout << test << "\n";

	return 0;

}
