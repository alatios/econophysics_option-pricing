#ifndef RNG__H
#define RNG__H

#include <iostream>
#include <cstdlib>
#include <cmath>		// log, cos, sin, ceil, M_PI

using namespace std;

//Abstract class Random Generator

class RNG_Generator{

	public:

		__device__ __host__ virtual double GetUniform() = 0;
		__device__ __host__ virtual double GetGauss() = 0;
		__device__ __host__ virtual void ResetSeed() = 0;

		__device__ __host__ virtual RNG_Generator& Set_internal_state(const unsigned int *) = 0;
};

//Linear congruential generator

class RNG_LCGS: public RNG_Generator{

	public:
		__device__ __host__ virtual void ResetSeed();
		__device__ __host__ virtual double GetUniform();
		__device__ __host__ virtual double GetGauss();

		__device__ __host__ virtual RNG_LCGS& Set_internal_state(const unsigned int *);

		__device__ __host__ RNG_LCGS();
		__device__ __host__ RNG_LCGS(const RNG_LCGS&);
		__device__ __host__ RNG_LCGS(unsigned int);
		__device__ __host__ ~RNG_LCGS() = default;

	private:
		unsigned int _seed;
		unsigned int _A = 1664525;
		unsigned int _C = 1013904223UL;

		__device__ __host__ unsigned int GetSeed() const;
		__device__ __host__ void SetSeed(unsigned int);

		friend class RNGCombinedGenerator;
};

//Tausworthe generator

class RNG_Tausworthe: public RNG_Generator{

	public:
		__device__ __host__ virtual void ResetSeed();
		__device__ __host__ virtual double GetUniform();
		__device__ __host__ virtual double GetGauss();

		__device__ __host__ virtual RNG_Tausworthe& Set_internal_state(const unsigned int *);

		__device__ __host__ RNG_Tausworthe();
		__device__ __host__ RNG_Tausworthe(const RNG_Tausworthe&);
		__device__ __host__ RNG_Tausworthe(unsigned int, int, int, int);
		__device__ __host__ ~RNG_Tausworthe() = default;

	private:
		unsigned int _seed;
		unsigned int _M;
		int _k1, _k2, _k3;

		__device__ __host__ unsigned int GetSeed() const;
		__device__ __host__ void SetSeed(unsigned int);

		__device__ __host__ unsigned int GetM() const;
		__device__ __host__ void SetM(unsigned int);
		__device__ __host__ int GetK1() const;
		__device__ __host__ void SetK1(int);
		__device__ __host__ int GetK2() const;
		__device__ __host__ void SetK2(int);
		__device__ __host__ int GetK3() const;
		__device__ __host__ void SetK3(int);

		friend class RNGCombinedGenerator;
};

//Combined generator

class RNGCombinedGenerator: public RNG_Generator{
	
	public:
		// Virtual functions from base class
		__device__ __host__ void ResetSeed();
		__device__ __host__ virtual double GetUniform();
		__device__ __host__ virtual double GetGauss();

		__device__ __host__ virtual RNG_Tausworthe& Set_internal_state(const unsigned int *);

		// Constructor and destructor
		__device__ __host__ RNGCombinedGenerator();
		__device__ __host__ RNGCombinedGenerator(const RNGCombinedGenerator&);
		__device__ __host__ RNGCombinedGenerator(const RNG_Tausworthe&, const RNG_LCGS&);
		__device__ __host__ ~RNGCombinedGenerator() = default;
	
	private:
		//3 taus + 1 LCGS
		RNG_Tausworthe _taus[3];
		RNG_LCGS _lcgs;

		__device__ __host__ RNG_Tausworthe& GetTausComponent(int) const;
		__device__ __host__ void SetTausComponent(int, const RNG_Tausworthe&);
		__device__ __host__ RNG_LCGS& GetLcgs() const;
		__device__ __host__ void SetLcgs(const RNG_LCGS&);

		// Single steps (callable from device and host, no need to diversify)
		__device__ __host__ unsigned LCGStep();
		__device__ __host__ unsigned TausStep1();
		__device__ __host__ unsigned TausStep2();
		__device__ __host__ unsigned TausStep3();
		__device__ __host__ double HybridTausGenerator();
};

#endif
