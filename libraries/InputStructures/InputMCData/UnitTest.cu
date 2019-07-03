#include <iostream>

#include "Input_MC_data.cuh"
#include "../InputGPUData/Input_gpu_data.cuh"

using namespace std;

int main(){

	Input_gpu_data gpu_data_1;
	Input_MC_data MC_data_1;
	Input_gpu_data gpu_data_2;
	Input_MC_data MC_data_2;
	Input_gpu_data gpu_data_3;
	Input_MC_data MC_data_3;

	gpu_data_1.NumberOfBlocks = 5;
	MC_data_1.NumberOfMCSimulations = 5120;
	gpu_data_2.NumberOfBlocks = 4;
	MC_data_2.NumberOfMCSimulations = 8190;
	gpu_data_3.NumberOfBlocks = 3;
	MC_data_3.NumberOfMCSimulations = 6150;


	cout << "\n-------------Input_MC_test-------------\n";
	cout << "\nMethod testing\n";

	bool test;
	test = (MC_data_1.GetNumberOfSimulationsPerThread(gpu_data_1) == static_cast<unsigned int>(2));
	cout << test << "\t";
	test = (MC_data_2.GetNumberOfSimulationsPerThread(gpu_data_2) == static_cast<unsigned int>(4));
	cout << test << "\t";
	test = (MC_data_3.GetNumberOfSimulationsPerThread(gpu_data_3) == static_cast<unsigned int>(5));
	cout << test << "\n";

	return 0;
}
