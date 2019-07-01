#include "Input_option_data.cuh"
#include <iostream>

using namespace std;
/*

OUTPUT:

  No problem ----> 1
  There is a problem ----> 0

*/

int main(){
/*
//Random choice only for testing
	double StrikePrice = 0.;
	unsigned int NumberOfIntervals = 1;
	double TimeToMaturity = 2.;
	char OptionType = 'p';
	bool test;

	Input_option_data OptionData_1 = Input_option_data();
	Input_option_data OptionData_2 = Input_option_data(StrikePrice, NumberOfIntervals, TimeToMaturity, OptionType);
	Input_option_data OptionData_3(OptionData_1);

	cout << "\n-------------Input_option_data_test-------------\n";
	cout << "Constructors testing\n";
	//Constructor testing
	test = (OptionData_1.GetStrikePrice()==static_cast<double>(110.)
	&& OptionData_1.GetNumberOfIntervals()==static_cast<unsigned int>(365)
	&& OptionData_1.GetTimeToMaturity()==static_cast<double>(365.)
	&& OptionData_1.GetOptionType()==static_cast<char>('c'))
	&& OptionData_1.GetDeltaTime()==static_cast<double>(1.);
	cout << test << "\t";
	test = (OptionData_2.GetStrikePrice()==static_cast<double>(0.)
	&& OptionData_2.GetNumberOfIntervals()==static_cast<unsigned int>(1)
	&& OptionData_2.GetTimeToMaturity()==static_cast<double>(2.)
	&& OptionData_2.GetOptionType()==static_cast<char>('p'))
	&& OptionData_2.GetDeltaTime()==static_cast<double>(2.);
	cout << test << "\t";
	test = (OptionData_3.GetStrikePrice()==static_cast<double>(110.)
	&& OptionData_3.GetNumberOfIntervals()==static_cast<unsigned int>(365)
	&& OptionData_3.GetTimeToMaturity()==static_cast<double>(365.)
	&& OptionData_3.GetOptionType()==static_cast<char>('c'))
	&& OptionData_3.GetDeltaTime()==static_cast<double>(1.);
	cout << test << "\n";

	cout << "\nMethods testing\n";
	//Methods testing
	OptionData_1.SetStrikePrice(3.);
	OptionData_1.SetNumberOfIntervals(4);
	OptionData_1.SetTimeToMaturity(5.);
	OptionData_1.SetOptionType('p');

	test = OptionData_1.GetStrikePrice()==static_cast<double>(3.);
	cout << test << "\t";
	test = OptionData_1.GetNumberOfIntervals()==static_cast<unsigned int>(4);
	cout << test << "\t";
	test = OptionData_1.GetTimeToMaturity()==static_cast<double>(5.);
	cout << test << "\t";
	test = OptionData_1.GetOptionType()==static_cast<char>('p');
	cout << test << "\t";
	test = OptionData_1.GetDeltaTime()==static_cast<double>(1.25);
	cout << test << "\n";
*/
	return 0;
}
