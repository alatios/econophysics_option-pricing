#ifndef _RNG__H
#define _RNG__H

#include <iostream>
#include <cstdlib>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

using namespace std;

class RNG{
	
	public:
		// Virtual functions from base class
		__device__ __host__ virtual void ResetSeed() = 0;
		__device__ __host__ virtual double GetUnsignedInt() = 0;
		__device__ __host__ virtual double GetUniform() = 0;
		__device__ __host__ virtual double GetGauss() = 0;
		__device__ __host__ virtual void SetInternalState(RNG*) = 0;
		
};

class RNG_CurandAdapter: public RNG{
	
	public:
	
		// Constructor and destructor
		__device__ __host__ RNG_CurandAdapter();
		__device__ __host__ RNG_CurandAdapter(const unsigned int seed, const unsigned int sequence);
		__device__ __host__ ~RNG_CurandAdapter() = default;
		
		// Virtual functions from base class
		__device__ __host__ void ResetSeed();
		__device__ __host__ double GetUnsignedInt();
		__device__ __host__ double GetUniform();
		__device__ __host__ double GetGauss();
		__device__ __host__ void SetInternalState(RNG*);
		
	private:
		
		curandState _State;
	
};

class RNG_CombinedGenerator: public RNG{
	
	public:
		// Virtual functions from base class
		__device__ __host__ void ResetSeed();
		__device__ __host__ double GetUnsignedInt();
		__device__ __host__ double GetUniform();
		__device__ __host__ double GetGauss();

		// Public internal state set (unsigned ints between 129 and UINT_MAX)
		__device__ __host__ void SetInternalState(RNG*);
		
		// Constructor and destructor
		__device__ __host__ RNG_CombinedGenerator();
		__device__ __host__ RNG_CombinedGenerator(const unsigned int, const unsigned int, const unsigned int, const unsigned int);
		__device__ __host__ ~RNG_CombinedGenerator() = default;
	
	private:
		// Seeds: 3 taus + 1 LCGS
		unsigned int _SeedLCGS;
		unsigned int _SeedTaus1;
		unsigned int _SeedTaus2;
		unsigned int _SeedTaus3;
		
		// Single steps (callable from device and host, no need to diversify)
		__device__ __host__ unsigned int LCGStep();
		__device__ __host__ unsigned int TausStep1();
		__device__ __host__ unsigned int TausStep2();
		__device__ __host__ unsigned int TausStep3();
		__device__ __host__ unsigned int HybridTausGenerator();
};

#endif
