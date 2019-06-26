#include "rng.cuh"

using namespace std;

__device__ __host__ RNG_CombinedGenerator::RNG_CombinedGenerator(const unsigned int seed1, const unsigned int seed2, const unsigned int seed3, const unsigned int seed4){
	this->_SeedLCGS = seed1;
	this->_SeedTaus1 = seed2;
	this->_SeedTaus2 = seed3;
	this->_SeedTaus3 = seed4;
}

__device__ __host__ RNG_CombinedGenerator::RNG_CombinedGenerator(){
	this->_SeedLCGS = 0;
	this->_SeedTaus1 = 129;
	this->_SeedTaus2 = 130;
	this->_SeedTaus3 = 131;
}

__device__ __host__ unsigned int RNG_CombinedGenerator::TausStep1(){
	int S1 = 13, S2 = 19, S3 = 12;
	unsigned int M = 4294967294UL;
	unsigned int z = this->_SeedTaus1;
	
	unsigned b = (((z << S1) ^ z) >> S2);
	
	this->_SeedTaus1 = (((z & M) << S3) ^ b);
	return _SeedTaus1;
}

__device__ __host__ unsigned int RNG_CombinedGenerator::TausStep2(){
	int S1 = 2, S2 = 25, S3 = 4;
	unsigned int M = 4294967288UL;
	unsigned int z = this->_SeedTaus2;
	
	unsigned b = (((z << S1) ^ z) >> S2);

	this->_SeedTaus2 = (((z & M) << S3) ^ b);
	return _SeedTaus2;
}
__device__ __host__ unsigned int RNG_CombinedGenerator::TausStep3(){
	int S1 = 3, S2 = 11, S3 = 17;
	unsigned int M = 4294967280UL;
	unsigned int z = this->_SeedTaus3;
	
	unsigned b = (((z << S1) ^ z) >> S2);

	this->_SeedTaus3 = (((z & M) << S3) ^ b);
	return _SeedTaus3;
}

__device__ __host__ unsigned int RNG_CombinedGenerator::LCGStep(){
	unsigned int z = this->_SeedLCGS;
	unsigned int A = 1664525;
	unsigned int C = 1013904223UL;
	this->_SeedLCGS = (A*z + C);
	return _SeedLCGS;
}

__device__ __host__ unsigned int RNG_CombinedGenerator::HybridTausGenerator(){
	return (
		this->TausStep1() ^
		this->TausStep2() ^
		this->TausStep3() ^
		this->LCGStep()
	);
}

__device__ __host__ void RNG_CombinedGenerator::ResetSeed(){
	this->_SeedLCGS = 0;
	this->_SeedTaus1 = 129;
	this->_SeedTaus2 = 130;
	this->_SeedTaus3 = 131;
}

__device__ __host__ double RNG_CombinedGenerator::GetUnsignedInt(){
	return this->HybridTausGenerator();
}

__device__ __host__ double RNG_CombinedGenerator::GetUniform(){
	return 2.3283064365387e-10 * this->GetUnsignedInt();
}

__device__ __host__ double RNG_CombinedGenerator::GetGauss(){
	double u = this->GetUniform();
	double v = this->GetUniform();

	return sqrt(-2.*log(u)) * cos(2.*M_PI*v);
}

__device__ __host__ void RNG_CombinedGenerator::SetInternalState(unsigned int seedLGCS, unsigned int seedTaus1, unsigned int seedTaus2, unsigned int seedTaus3){
	this->_SeedLCGS = seedLGCS;
	this->_SeedTaus1 = seedTaus1;
	this->_SeedTaus2 = seedTaus2;
	this->_SeedTaus3 = seedTaus3;
}
