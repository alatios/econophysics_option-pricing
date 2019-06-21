#include "Statistics.cuh"
#include <iostream>

using namespace std;
/*

OUTPUT:

  No problem ----> 1
  There is a problem ----> 0

*/

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
	cout << test << "\t";
	test = statistics.GetSquaredPayoffCounter() == static_cast<unsigned int>(0);
	cout << test << "\n";

	cout << "\nMethods testing\n";

	statistics.AddToPayoffSum(5.);
	statistics.AddToSquaredPayoffSum(25.);

	test = statistics.GetPayoffSum() == static_cast<double>(5.);
	cout << test << "\t";
	test = statistics.GetSquaredPayoffSum() == static_cast<double>(25.);
	cout << test << "\t";
	test = statistics.GetPayoffCounter() == static_cast<unsigned int>(1);
	cout << test << "\t";
	test = statistics.GetSquaredPayoffCounter() == static_cast<unsigned int>(1);
	cout << test << "\n";

	statistics.ResetPayoffSum();
	statistics.ResetSquaredPayoffSum();
	statistics.AddToAll(3.);
	statistics.AddToAll(1.);

	test = statistics.GetPayoffSum() == static_cast<double>(4.);
	cout << test << "\t";
	test = statistics.GetSquaredPayoffSum() == static_cast<double>(10.);
	cout << test << "\t";
	test = statistics.GetPayoffCounter() == static_cast<unsigned int>(2);
	cout << test << "\t";
	test = statistics.GetSquaredPayoffCounter() == static_cast<unsigned int>(2);
	cout << test << "\n";

	statistics.ResetAllSums();

	test = statistics.GetPayoffSum() == static_cast<double>(0.);
	cout << test << "\t";
	test = statistics.GetSquaredPayoffSum() == static_cast<double>(0.);
	cout << test << "\t";
	test = statistics.GetPayoffCounter() == static_cast<unsigned int>(0);
	cout << test << "\t";
	test = statistics.GetSquaredPayoffCounter() == static_cast<unsigned int>(0);
	cout << test << "\n";

	return 0;
}
