#ifndef _DATA__STREAM__MANAGER_H_
#define _DATA__STREAM__MANAGER_H_

#include <string>	// string

#include "../../InputStructures/InputGPUData/Input_gpu_data.cuh"
#include "../../InputStructures/InputMarketData/Input_market_data.cuh"
#include "../../InputStructures/InputMCData/Input_MC_data.cuh"
#include "../../InputStructures/InputOptionData/Input_option_data.cuh"
#include "../Statistics/Statistics.cuh"
#include "../../OutputStructures/OutputMCData/Output_MC_data.cuh"

class Data_stream_manager{
	
	private:
	
		std::string _InputFile;
	
	public:
	
		__host__ Data_stream_manager();
		__host__ Data_stream_manager(std::string inputFile);
		__host__ ~Data_stream_manager() = default;
		
		// Set input file
		__host__ void SetInputFile(std::string);
	
		// Input processing
		__host__ void ReadInputData(Input_gpu_data&, Input_option_data&, Input_market_data&, Input_MC_data&) const;
		__host__ void PrintInputData(const Input_gpu_data&, const Input_option_data&, const Input_market_data&, const Input_MC_data&) const;

		// Output processing
		__host__ void StoreOutputData(Output_MC_data&, const Statistics exactResults, const Statistics eulerResults, double elapsedTime, char hostOrDevice) const;
		__host__ void PrintOutputData(const Output_MC_data&) const;		
		
	
};

#endif
