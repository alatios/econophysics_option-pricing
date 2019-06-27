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

// Payoff evaluation
__host__ __device__ double EvaluatePayoff(const Path&, const Input_option_data&);
__host__ __device__ double ActualizePayoff(double payoff, double riskFreeRate, double timeToMaturity);

// Input processing
__host__ void PrintInputData(const Input_gpu_data&, const Input_option_data&, const Input_market_data&, const Input_MC_data&);
__host__ void ReadInputData(vector<string>&, string sourceFile);

// Output processing
__host__ void PrintOutputData(const Output_MC_data&);

#endif
