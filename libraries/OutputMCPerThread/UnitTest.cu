#include "Output_MC_per_thread.cuh"
#include <iostream>

using namespace std;
/*

OUTPUT:

  No problem ----> 1
  There is a problem ----> 0

*/

int main(){

	bool test;

	cout << endl << "-------------Output_MC_per_thread_test-------------" << endl;
	cout << "Constructors testing" << endl;

	Output_MC_per_thread out_per_thread;

	test = out_per_thread.GetPayoffSum() == static_cast<double>(0.);
	cout << test << "\t";
	test = out_per_thread.GetSquaredPayoffSum() == static_cast<double>(0.);
	cout << test << "\t";
	test = out_per_thread.GetPayoffCounter() == static_cast<unsigned int>(0);
	cout << test << "\t";
	test = out_per_thread.GetSquaredPayoffCounter() == static_cast<unsigned int>(0);
	cout << test << "\n\n\n";

	cout << "Methods testing" << endl;

	out_per_thread.AddToPayoffSum(5.);
	out_per_thread.AddToSquaredPayoffSum(25.);

	test = out_per_thread.GetPayoffSum() == static_cast<double>(5.);
	cout << test << "\t";
	test = out_per_thread.GetSquaredPayoffSum() == static_cast<double>(25.);
	cout << test << "\t";
	test = out_per_thread.GetPayoffCounter() == static_cast<unsigned int>(1);
	cout << test << "\t";
	test = out_per_thread.GetSquaredPayoffCounter() == static_cast<unsigned int>(1);
	cout << test << "\n\n";

	out_per_thread.ResetPayoffSum();
	out_per_thread.ResetSquaredPayoffSum();
	out_per_thread.AddToAll(3.);
	out_per_thread.AddToAll(1.);

	test = out_per_thread.GetPayoffSum() == static_cast<double>(4.);
	cout << test << "\t";
	test = out_per_thread.GetSquaredPayoffSum() == static_cast<double>(10.);
	cout << test << "\t";
	test = out_per_thread.GetPayoffCounter() == static_cast<unsigned int>(2);
	cout << test << "\t";
	test = out_per_thread.GetSquaredPayoffCounter() == static_cast<unsigned int>(2);
	cout << test << "\n\n";

	out_per_thread.ResetAllSums();

	test = out_per_thread.GetPayoffSum() == static_cast<double>(0.);
	cout << test << "\t";
	test = out_per_thread.GetSquaredPayoffSum() == static_cast<double>(0.);
	cout << test << "\t";
	test = out_per_thread.GetPayoffCounter() == static_cast<unsigned int>(0);
	cout << test << "\t";
	test = out_per_thread.GetSquaredPayoffCounter() == static_cast<unsigned int>(0);
	cout << test << "\n\n";

	return 0;
}
