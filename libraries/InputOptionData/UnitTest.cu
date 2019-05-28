#include "Input_option_data.cuh"
#include <iostream>

using namespace std;
/*

OUTPUT:

  No problem ----> 1
  There is a problem ----> 0

*/

int main(){

//Random choice only for testing
  float StrikePrice = 0.;
  unsigned int NumberOfIntervals = 1;
  float TimeToMaturity = 2.;
  char OptionType = 'p';
  bool test;

	Input_option_data OptionData_1 = Input_option_data();
	Input_option_data OptionData_2 = Input_option_data(StrikePrice, NumberOfIntervals, TimeToMaturity, OptionType);
	Input_option_data OptionData_3(OptionData_1);

	cout << endl << "-------------Input_option_data_test-------------" << endl;
	cout << "Constructors testing" << endl;
	//Constructor testing
	test = (OptionData_1.GetStrikePrice()==static_cast<float>(110.)
	&& OptionData_1.GetNumberOfIntervals()==static_cast<unsigned int>(365)
	&& OptionData_1.GetTimeToMaturity()==static_cast<float>(365.)
	&& OptionData_1.GetDeltaTime()==static_cast<float>(1.)
	&& OptionData_1.GetOptionType()==static_cast<char>('c'));
	cout << test << endl;
	test = (OptionData_2.GetStrikePrice()==static_cast<float>(0.)
	&& OptionData_2.GetNumberOfIntervals()==static_cast<unsigned int>(1)
	&& OptionData_2.GetTimeToMaturity()==static_cast<float>(2.)
	&& OptionData_2.GetDeltaTime()==static_cast<float>(2.)
	&& OptionData_2.GetOptionType()==static_cast<char>('p'));
	cout << test << endl;
	test = (OptionData_3.GetStrikePrice()==static_cast<float>(110.)
	&& OptionData_3.GetNumberOfIntervals()==static_cast<unsigned int>(365)
	&& OptionData_3.GetTimeToMaturity()==static_cast<float>(365.)
	&& OptionData_3.GetDeltaTime()== static_cast<float>(1.)
	&& OptionData_3.GetOptionType()==static_cast<char>('c'));
	cout << test << endl;

	cout << "Methods testing" << endl;
	//Methods testing
	OptionData_1.SetStrikePrice(3.);
	OptionData_1.SetNumberOfIntervals(4);
	OptionData_1.SetTimeToMaturity(5.);
	OptionData_1.SetOptionType('p');

	test = OptionData_1.GetStrikePrice()==static_cast<float>(3.);
	cout << test << endl;
	test = OptionData_1.GetNumberOfIntervals()==static_cast<unsigned int>(4);
	cout << test << endl;
	test = OptionData_1.GetTimeToMaturity()==static_cast<float>(5.);
	cout << test << endl;
	test = OptionData_1.GetDeltaTime()==static_cast<float>(static_cast<float>(5.) / static_cast<unsigned int>(4));
	cout << test << endl;
	test = OptionData_1.GetOptionType()==static_cast<char>('p');
	cout << test << endl;

	return 0;
}
