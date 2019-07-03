#ifndef _SUPPORT_FUNCTIONS_H_
#define _SUPPORT_FUNCTIONS_H_

#include <vector>	// vector
#include <string>	// string

#include "../../InputStructures/InputGPUData/Input_gpu_data.cuh"
#include "../../InputStructures/InputMarketData/Input_market_data.cuh"
#include "../../InputStructures/InputMCData/Input_MC_data.cuh"
#include "../../InputStructures/InputOptionData/Input_option_data.cuh"
#include "../DataStreamManager/Data_stream_manager.cuh"
#include "../Statistics/Statistics.cuh"

// Main evaluators (host-device paradigm)
__host__ void OptionPricingEvaluator_Host(Input_gpu_data, Input_option_data, Input_market_data, Input_MC_data, Statistics* exactOutputs, Statistics* eulerOutputs, unsigned int seed);
__host__ __device__ void OptionPricingEvaluator_HostDev(Input_gpu_data, Input_option_data, Input_market_data, Input_MC_data, Statistics* exactOutputs, Statistics* eulerOutputs, unsigned int seed, unsigned int threadNumber);
__global__ void OptionPricingEvaluator_Global(Input_gpu_data, Input_option_data, Input_market_data, Input_MC_data, Statistics* exactOutputs, Statistics* eulerOutputs, unsigned int seed);

// CPU and GPU algorithms
__host__ void CPUOptionPricingMonteCarloAlgorithm(Data_stream_manager, Input_gpu_data, Input_option_data, Input_market_data, Input_MC_data, unsigned int seed);
__host__ void GPUOptionPricingMonteCarloAlgorithm(Data_stream_manager, Input_gpu_data, Input_option_data, Input_market_data, Input_MC_data, unsigned int seed);

#endif
