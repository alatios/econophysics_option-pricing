#ifndef _SUPPORT_FUNCTIONS_H_
#define _SUPPORT_FUNCTIONS_H_

#include <iostream>
#include <vector>	// vector
#include <string>	// string

#include "../../InputStructures/InputGPUData/Input_gpu_data.cuh"
#include "../../InputStructures/InputMarketData/Input_market_data.cuh"
#include "../../InputStructures/InputMCData/Input_MC_data.cuh"
#include "../../InputStructures/InputOptionData/Input_option_data.cuh"
#include "../Path/Path.cuh"
#include "../Statistics/Statistics.cuh"
#include "../RandomGenerator/rng.cuh"
#include "../../OutputStructures/OutputMCData/Output_MC_data.cuh"

// Main evaluators (host-device paradigm)
__host__ void OptionPricingEvaluator_Host(Input_gpu_data, Input_option_data, Input_market_data, Input_MC_data, Statistics* exactOutputs, Statistics* eulerOutputs);
__host__ __device__ void OptionPricingEvaluator_HostDev(Input_gpu_data, Input_option_data, Input_market_data, Input_MC_data, Statistics* exactOutputs, Statistics* eulerOutputs, unsigned int threadNumber);
__global__ void OptionPricingEvaluator_Global(Input_gpu_data, Input_option_data, Input_market_data, Input_MC_data, Statistics* exactOutputs, Statistics* eulerOutputs);

#endif
