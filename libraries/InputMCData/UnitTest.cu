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
	unsigned int NumberOfMCSimulationsPerThread = 30;
	bool test;
	
	Input_gpu_data GpuData(6, 7);
	
	Input_MC_data MCData_1;
	Input_MC_data MCData_2 = Input_MC_data(NumberOfMCSimulationsPerThread);
	Input_MC_data MCData_3 = Input_MC_data(GpuData);
	Input_MC_data MCData_4 = Input_MC_data(NumberOfMCSimulationsPerThread, GpuData);
	
	cout << endl << "-------------Input_MC_data_test-------------" << endl;
	cout << "Constructors testing" << endl;
	test = MCData_1.GetNumberOfMCSimulations()==static_cast<unsigned int>(5*14336);
	cout << test << "\t";
	test = MCData_1.GetNumberOfMCSimulationsPerThread()==static_cast<unsigned int>(5);
	cout << test << "\t";
	test = MCData_1.GetGpuData().GetNumberOfBlocks()==static_cast<unsigned int>(14);
	cout << test << "\t";
	test = MCData_1.GetGpuData().GetNumberOfThreadsPerBlock() == static_cast<unsigned int>(1024);
	cout << test << endl;
	
	test = MCData_2.GetNumberOfMCSimulations()==static_cast<unsigned int>(30*14336);
	cout << test << "\t";
	test = MCData_2.GetNumberOfMCSimulationsPerThread()==static_cast<unsigned int>(30);
	cout << test << "\t";
	test = MCData_2.GetGpuData().GetNumberOfBlocks()==static_cast<unsigned int>(14);
	cout << test << "\t";
	test = MCData_2.GetGpuData().GetNumberOfThreadsPerBlock() == static_cast<unsigned int>(1024);
	cout << test << endl;
	
	test = MCData_3.GetNumberOfMCSimulations()==static_cast<unsigned int>(5*42);
	cout << test << "\t";
	test = MCData_3.GetNumberOfMCSimulationsPerThread()==static_cast<unsigned int>(5);
	cout << test << "\t";
	test = MCData_3.GetGpuData().GetNumberOfBlocks()==static_cast<unsigned int>(6);
	cout << test << "\t";
	test = MCData_3.GetGpuData().GetNumberOfThreadsPerBlock() == static_cast<unsigned int>(7);
	cout << test << endl;
	
	test = MCData_4.GetNumberOfMCSimulations()==static_cast<unsigned int>(30*42);
	cout << test << "\t";
	test = MCData_4.GetNumberOfMCSimulationsPerThread()==static_cast<unsigned int>(30);
	cout << test << "\t";
	test = MCData_4.GetGpuData().GetNumberOfBlocks()==static_cast<unsigned int>(6);
	cout << test << "\t";
	test = MCData_4.GetGpuData().GetNumberOfThreadsPerBlock() == static_cast<unsigned int>(7);
	cout << test << endl;	
	
	cout << endl << endl; 
	
	cout << "Methods testing" << endl;
	MCData_1.SetNumberOfMCSimulationsPerThread(50);
	Input_gpu_data OtherGpuData(55, 90);
	MCData_1.SetGpuData(OtherGpuData);
	test = MCData_1.GetNumberOfMCSimulations()==static_cast<unsigned int>(50*90*55);
	cout << test << "\t";
	test = MCData_1.GetNumberOfMCSimulationsPerThread()==static_cast<unsigned int>(50);
	cout << test << "\t";
	test = MCData_1.GetGpuData().GetNumberOfBlocks()==static_cast<unsigned int>(55);
	cout << test << "\t";
	test = MCData_1.GetGpuData().GetNumberOfThreadsPerBlock() == static_cast<unsigned int>(90);
	cout << test << endl;

	return 0;
}
