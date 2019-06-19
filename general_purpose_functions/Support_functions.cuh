#ifndef _SUPPORT_FUNCTIONS_H_
#define _SUPPORT_FUNCTIONS_H_

#include <iostream>
#include <vector>	// vector
#include <tuple>	// tuple
#include <string>	// string

#include "../libraries/InputGPUData/Input_gpu_data.cuh"
#include "../libraries/InputMarketData/Input_market_data.cuh"
#include "../libraries/InputMCData/Input_MC_data.cuh"
#include "../libraries/InputOptionData/Input_option_data.cuh"
#include "../libraries/Path/Path.cuh"
#include "../libraries/OutputMCPerThread/Output_MC_per_thread.cuh"
#include "../random_generator/rng.cuh"

// Main evaluators (host-device paradigm)
__host__ void OptionPricingEvaluator_Host(Input_gpu_data, Input_option_data, Input_market_data, Input_MC_data, Path, RNGCombinedGenerator*, Output_MC_per_thread*);
__host__ __device__ void OptionPricingEvaluator_HostDev(Input_gpu_data, Input_option_data, Input_market_data, Input_MC_data, Path, RNGCombinedGenerator*, Output_MC_per_thread*, unsigned int threadNumber);
__global__ void OptionPricingEvaluator_Global(Input_gpu_data, Input_option_data, Input_market_data, Input_MC_data, Path, RNGCombinedGenerator*, Output_MC_per_thread*);

// Payoff evaluation
__host__ __device__ double EvaluatePayoff(const Path&, const Input_option_data&);
__host__ __device__ double ActualizePayoff(double payoff, double riskFreeRate, double timeToMaturity);

// Input processing
__host__ void PrintInputData(const Input_gpu_data&, const Input_option_data&, const Input_market_data&, const Input_MC_data&);
__host__ void ReadInputData(vector<string>&, string sourceFile);

// Output processing
__host__ tuple<double, double> EvaluateEstimatedPriceAndError(Output_MC_per_thread*, unsigned int totalNumberOfThreads);

#endif
