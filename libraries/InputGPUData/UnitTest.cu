#include "Input_gpu_data.cuh"
#include <iostream>

using namespace std;
/*

OUTPUT:

  No problem ----> 1
  There is a problem ----> 0

*/

int main(){

	//Random choice only for testing
	unsigned int NumberOfBlocks = 100;
	bool test;

	Input_gpu_data GpuData_1 = Input_gpu_data();
	Input_gpu_data GpuData_2 = Input_gpu_data(NumberOfBlocks);
	Input_gpu_data GpuData_3(GpuData_1);

	cout << endl << "-------------Input_gpu_test-------------" << endl;
	cout << "Constructors testing" << endl;

	test = (GpuData_1.GetNumberOfBlocks()==static_cast<unsigned int>(14));
	cout << test << endl;
	test = (GpuData_2.GetNumberOfBlocks()==static_cast<unsigned int>(100));
	cout << test << endl;
	test = (GpuData_3.GetNumberOfBlocks()==static_cast<unsigned int>(14));
	cout << test << endl;

	cout << "Methods testing" << endl;
	//Methods testing
	GpuData_1.SetNumberOfBlocks(6);

	test = GpuData_1.GetNumberOfBlocks()==static_cast<unsigned int>(6);
	cout << test << endl;

	return 0;
}
