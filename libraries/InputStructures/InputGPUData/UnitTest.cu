#include "Input_gpu_data.cuh"
#include <iostream>

using namespace std;
/*

OUTPUT:

  No problem ----> 1
  There is a problem ----> 0

*/

int main(){

/*	//Random choice only for testing
	unsigned int NumberOfBlocks = 100;
	bool test;

	Input_gpu_data GpuData_1 = Input_gpu_data();
	Input_gpu_data GpuData_2 = Input_gpu_data(NumberOfBlocks);
	Input_gpu_data GpuData_3(GpuData_1);

	cout << "\n-------------Input_gpu_test-------------\n";
	cout << "Constructors testing\n";

	test = (GpuData_1.GetNumberOfBlocks()==static_cast<unsigned int>(14));
	cout << test << "\t";
	test = (GpuData_2.GetNumberOfBlocks()==static_cast<unsigned int>(100));
	cout << test << "\t";
	test = (GpuData_3.GetNumberOfBlocks()==static_cast<unsigned int>(14));
	cout << test << "\n";

	cout << "\nMethods testing\n";
	//Methods testing
	GpuData_1.SetNumberOfBlocks(6);

	test = GpuData_1.GetNumberOfBlocks()==static_cast<unsigned int>(6);
	cout << test << "\t";
	test = GpuData_1.GetNumberOfThreadsPerBlock()==static_cast<unsigned int>(512);
	cout << test << "\t";
	test = GpuData_1.GetTotalNumberOfThreads()==static_cast<unsigned int>(3072);
	cout << test << "\n";

*/
	return 0;
}
