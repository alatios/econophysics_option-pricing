#ifndef _RNG__H
#define _RNG__H

#include <iostream>
#include <cstdlib>
#include <cmath>		// log, cos, sin, ceil, M_PI

using namespace std;

class RNG{
	
	public:
		// Virtual functions from base class
		__device__ __host__ virtual void ResetSeed() = 0;
		__device__ __host__ virtual double GetUnsignedInt() = 0;
		__device__ __host__ virtual double GetUniform() = 0;
		__device__ __host__ virtual double GetGauss() = 0;
		__device__ __host__ virtual void SetInternalState(unsigned int, unsigned int, unsigned int, unsigned int) = 0;
		
};

class RNG_CombinedGenerator: public RNG{
	
	public:
		// Virtual functions from base class
		__device__ __host__ void ResetSeed();
		__device__ __host__ double GetUnsignedInt();
		__device__ __host__ double GetUniform();
		__device__ __host__ double GetGauss();

		// Public internal state set (unsigned ints between 129 and UINT_MAX)
		__device__ __host__ void SetInternalState(unsigned int, unsigned int, unsigned int, unsigned int);
		
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
