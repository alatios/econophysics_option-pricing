#include <iostream>

#include "Statistics.cuh"

using namespace std;

int main(){

	bool test;

	cout << "\n-------------Statistics_test-------------\n";
	cout << "Constructors testing\n";

	Statistics statistics;

	test = statistics.GetPayoffSum() == static_cast<double>(0.);
	cout << test << "\t";
	test = statistics.GetSquaredPayoffSum() == static_cast<double>(0.);
	cout << test << "\t";
	test = statistics.GetPayoffCounter() == static_cast<unsigned int>(0);
	cout << test << "\n";

	cout << "\nMethods testing\n";

	statistics.AddPayoff(5.);

	test = statistics.GetPayoffSum() == static_cast<double>(5.);
	cout << test << "\t";
	test = statistics.GetSquaredPayoffSum() == static_cast<double>(25.);
	cout << test << "\t";
	test = statistics.GetPayoffCounter() == static_cast<unsigned int>(1);
	cout << test << "\n";

	statistics.ResetSums();

	test = statistics.GetPayoffSum() == static_cast<double>(0.);
	cout << test << "\t";
	test = statistics.GetSquaredPayoffSum() == static_cast<double>(0.);
	cout << test << "\t";
	test = statistics.GetPayoffCounter() == static_cast<unsigned int>(0);
	cout << test << "\n";

	statistics.AddPayoff(2.);
	statistics.AddPayoff(2.);
	statistics.EvaluateEstimatedPriceAndError();

	test = statistics.GetPayoffAverage() == static_cast<double>(2.);
	cout << test << "\t";
	test = statistics.GetPayoffError() == static_cast<double>(0.);
	cout << test << "\n";

	statistics.ResetSums();
	statistics.AddPayoff(2.);

	Statistics statistics_1;
	statistics_1.AddPayoff(5.);

	Statistics statistics_sum;

	statistics_sum = statistics += statistics_1;

	test = statistics_sum.GetPayoffSum() == static_cast<double>(7.);
	cout << test << "\t";
	test = statistics_sum.GetSquaredPayoffSum() == static_cast<double>(29.);
	cout << test << "\t";
	test = statistics_sum.GetPayoffCounter() == static_cast<unsigned int>(2);
	cout << test << "\n";

	return 0;
}
