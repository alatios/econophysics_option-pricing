#ifndef RNG__H
#define RNG__H

#include <iostream>
#include <cstdlib>
#include <cmath>		// log, cos, sin, ceil, M_PI

using namespace std;

class RNGCombinedGenerator{
	
	public:
		// Virtual functions from base class
		__device__ __host__ void ResetSeed();
		__device__ __host__ float GetUniform();
		__device__ __host__ float GetGauss();
		
		// Single steps (callable from device and host, no need to diversify)
		__device__ __host__ unsigned LCGStep();
		__device__ __host__ unsigned TausStep1();
		__device__ __host__ unsigned TausStep2();
		__device__ __host__ unsigned TausStep3();
		__device__ __host__ float HybridTausGenerator();
		
		// Get/set seeds
		__device__ __host__ unsigned GetSeedLCGS() const;
		__device__ __host__ void SetSeedLCGS(const unsigned int);
		__device__ __host__ unsigned GetSeedTaus1() const;
		__device__ __host__ void SetSeedTaus1(const unsigned int);
		__device__ __host__ unsigned GetSeedTaus2() const;
		__device__ __host__ void SetSeedTaus2(const unsigned int);
		__device__ __host__ unsigned GetSeedTaus3() const;
		__device__ __host__ void SetSeedTaus3(const unsigned int);
		
		// Constructor and destructor
		__device__ __host__ RNGCombinedGenerator();
		__device__ __host__ RNGCombinedGenerator(const RNGCombinedGenerator&);
		__device__ __host__ RNGCombinedGenerator(const unsigned int, const unsigned int, const unsigned int, const unsigned int);
		__device__ __host__ ~RNGCombinedGenerator() = default;
	
	private:
		// Seeds: 3 taus + 1 LCGS
		unsigned int _seedLCGS;
		unsigned int _seedTaus1;
		unsigned int _seedTaus2;
		unsigned int _seedTaus3;
};

#endif
