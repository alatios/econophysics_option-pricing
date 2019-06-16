#ifndef RNG__H
#define RNG__H

#include <iostream>
#include <cstdlib>
#include <random>		// C++11 Mersenne twister
#include <cmath>		// log, cos, sin, ceil, M_PI

using namespace std;

class RandomNumberGenerator{
	
	public:
		
		__device__ __host__ virtual unsigned int GetUnsignedInt() = 0;
		__device__ __host__ virtual double GetUniform() = 0;
		__device__ __host__ virtual double GetGauss() = 0;
		__device__ __host__ virtual void ResetSeed() = 0;
		
		__device__ __host__ virtual void SetInternalState(RandomNumberGenerator *supportGenerator);
	
};

class RandomNumberGenerator_Mersenne: public RandomNumberGenerator{
	
	public:
	
		__device__ __host__ RandomNumberGenerator_Mersenne();
		__device__ __host__ RandomNumberGenerator_Mersenne(unsigned int seed);
		__device__ __host__ ~RandomNumberGenerator_Mersenne();
		
		__device__ __host__ unsigned int GetUnsignedInt();
		__device__ __host__ double GetUniform();
		__device__ __host__ double GetGauss();
		__device__ __host__ void ResetSeed();
		
		__device__ __host__ void SetInternalState(RandomNumberGenerator *supportGenerator);
		__device__ __host__ void SetInternalState(unsigned int seed);
		
	private:
	
		mt19937* m_CoreMersenneGenerator;
	
};
/*
class RandomNumberGenerator_Hybrid: public RandomNumberGenerator{
	
	public:
	
		__device__ __host__ RandomNumberGenerator_Mersenne();
		__device__ __host__ RandomNumberGenerator_Mersenne(unsigned int seedLGCS, unsigned int seedTaus1, unsigned int seedTaus2, unsigned int seedTaus3);
		__device__ __host__ ~RandomNumberGenerator_Mersenne();
		
		__device__ __host__ unsigned int GetUnsignedInt();
		__device__ __host__ double GetUniform();
		__device__ __host__ double GetGauss();
		__device__ __host__ void ResetSeed();
		
		__device__ __host__ void SetInternalState(RandomNumberGenerator *supportGenerator);
		__device__ __host__ void SetInternalState(unsigned int seedLGCS, unsigned int seedTaus1, unsigned int seedTaus2, unsigned int seedTaus3);
		
	private:
	
		unsigned int m_seedLGCS;
		unsigned int m_seedTaus1;
		unsigned int m_seedTaus2;
		unsigned int m_seedTaus3;
		
		__device__ __host__ unsigned int LCGStep();
		__device__ __host__ unsigned int TausStep1();
		__device__ __host__ unsigned int TausStep2();
		__device__ __host__ unsigned int TausStep3();
		__device__ __host__ double HybridGenerator();
	
};
*/
#endif
