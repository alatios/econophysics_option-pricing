#include "Input_option_data.cuh"
#include <iostream>

using namespace std;


int main(){


	Input_option_data option_data_1;
	Input_option_data option_data_2;
	Input_option_data option_data_3;



	option_data_1.OptionType = 'c';
	option_data_1.NumberOfIntervals = 10;
	option_data_1.TimeToMaturity = 1.;
	option_data_1.StrikePrice = 100.;

	option_data_2.OptionType = 'f';
	option_data_2.NumberOfIntervals = 10;
	option_data_2.TimeToMaturity = 1.;
	option_data_2.StrikePrice = 100.;

	option_data_3.OptionType = 'e';
	option_data_3.NumberOfIntervals = 10;
	option_data_3.TimeToMaturity = 1.;
	option_data_3.B = 1.;
	option_data_3.N = 2.;
	option_data_3.K = 3.;


	cout << "\n-------------Input_option_test-------------\n";
	cout << "\nMethods testing\n";

	bool test;
	test = (option_data_1.OptionType == char('c'));
	cout << test << "\t";
	test = (option_data_1.NumberOfIntervals == static_cast<unsigned int>(10));
	cout << test << "\t";
	test = (option_data_1.TimeToMaturity == static_cast<double>(1.));
	cout << test << "\t";
	test = (option_data_1.StrikePrice == static_cast<double>(100.));
	cout << test << "\t";
	test = (option_data_1.GetDeltaTime() == static_cast<double>(0.1));
	cout << test << "\n";

	test = (option_data_2.OptionType == char('f'));
	cout << test << "\n";

	test = (option_data_3.OptionType == char('e'));
	cout << test << "\t";
	test = (option_data_3.B == static_cast<double>(1.));
	cout << test << "\t";
	test = (option_data_3.N == static_cast<double>(2.));
	cout << test << "\t";
	test = (option_data_3.K == static_cast<double>(3.));
	cout << test << "\n";

	return 0;
}
