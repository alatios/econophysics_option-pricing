#ifndef RNG__H
#define RNG__H

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
		
};

class RNGCombinedGenerator{
	
	public:
		// Virtual functions from base class
		__device__ __host__ void ResetSeed();
		__device__ __host__ double GetUnsignedInt();
		__device__ __host__ double GetUniform();
		__device__ __host__ double GetGauss();

		// Public internal state set (unsigned ints between 129 and UINT_MAX)
		__device__ __host__ void SetInternalState(unsigned int, unsigned int, unsigned int, unsigned int);
		
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
		
		// Get/set seeds
		__device__ __host__ unsigned int GetSeedLCGS() const;
		__device__ __host__ void SetSeedLCGS(const unsigned int);
		__device__ __host__ unsigned int GetSeedTaus1() const;
		__device__ __host__ void SetSeedTaus1(const unsigned int);
		__device__ __host__ unsigned int GetSeedTaus2() const;
		__device__ __host__ void SetSeedTaus2(const unsigned int);
		__device__ __host__ unsigned int GetSeedTaus3() const;
		__device__ __host__ void SetSeedTaus3(const unsigned int);
		
		// Single steps (callable from device and host, no need to diversify)
		__device__ __host__ unsigned int LCGStep();
		__device__ __host__ unsigned int TausStep1();
		__device__ __host__ unsigned int TausStep2();
		__device__ __host__ unsigned int TausStep3();
		__device__ __host__ unsigned int HybridTausGenerator();
};

#endif
