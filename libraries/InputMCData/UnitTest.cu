#include "Input_MC_data.cuh"
#include <iostream>

using namespace std;
/*

OUTPUT:

  No problem ----> 1
  There is a problem ----> 0

*/

int main(){
	
	//Random choice only for testing
	unsigned int NumberOfMCSimulations = 1000000;
	bool test;
	
	Input_MC_data MCData_1;
	Input_MC_data MCData_2 = Input_MC_data(NumberOfMCSimulations);
	
	cout << "\n-------------Input_MC_data_test-------------\n";
	cout << "Constructors testing\n";
	test = MCData_1.GetNumberOfMCSimulations()==static_cast<unsigned int>(5000000);
	cout << test << "\t";
	
	test = MCData_2.GetNumberOfMCSimulations()==static_cast<unsigned int>(1000000);
	cout << test << "\n";
	
	cout << "\nMethods testing\n" << endl;
	MCData_1.SetNumberOfMCSimulations(2000000);
	Input_gpu_data gpuData;
	test = MCData_1.GetNumberOfMCSimulations()==static_cast<unsigned int>(2000000);
	cout << test << "\t";
	test = MCData_1.GetNumberOfSimulationsPerThread(gpuData)==static_cast<unsigned int>(280);
	cout << test << "\n";

	return 0;
}
