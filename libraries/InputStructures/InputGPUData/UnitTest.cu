#include <iostream>

#include "Input_gpu_data.cuh"

using namespace std;

int main(){

	Input_gpu_data gpu_data_1;

	gpu_data_1.NumberOfBlocks = 100;

	cout << "\n-------------Input_gpu_test-------------\n";
	cout << "\nMethods testing\n";

	bool test;
	test = (gpu_data_1.GetNumberOfThreadsPerBlock() == static_cast<unsigned int>(512));
	cout << test << "\t";
	test = (gpu_data_1.GetTotalNumberOfThreads() == static_cast<unsigned int>(51200));
	cout << test << "\n";

	return 0;
}
