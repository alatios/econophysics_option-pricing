#include "rng.cuh"

using namespace std;

//
//	RandomNumberGenerator_Hybrid
//

// Constructors
__device__ __host__ RandomNumberGenerator_Hybrid::RandomNumberGenerator_Hybrid(){
	this->SetSeedLCGS(static_cast<unsigned int>(0));
	this->SetSeedTaus1(static_cast<unsigned int>(129));
	this->SetSeedTaus2(static_cast<unsigned int>(130));
	this->SetSeedTaus3(static_cast<unsigned int>(131));
}
__device__ __host__ RandomNumberGenerator_Hybrid::RandomNumberGenerator_Hybrid(unsigned int seedLGCS, unsigned int seedTaus1, unsigned int seedTaus2, unsigned int seedTaus3){
	this->SetSeedLCGS(seedLGCS);
	this->SetSeedTaus1(seedTaus1);
	this->SetSeedTaus2(seedTaus2);
	this->SetSeedTaus3(seedTaus3);
}

// Private set/get methods
__device__ __host__ unsigned int RandomNumberGenerator_Hybrid::GetSeedLCGS(){
	return m_seedLGCS;
}

__device__ __host__ void RandomNumberGenerator_Hybrid::SetSeedLCGS(unsigned int seed){
	m_seedLGCS = seed;
}

__device__ __host__ unsigned int RandomNumberGenerator_Hybrid::GetSeedTaus1(){
	return m_seedTaus1;
}

__device__ __host__ void RandomNumberGenerator_Hybrid::SetSeedTaus1(unsigned int seed){
	m_seedTaus1 = seed;
}

__device__ __host__ unsigned int RandomNumberGenerator_Hybrid::GetSeedTaus2(){
	return m_seedTaus2;
}

__device__ __host__ void RandomNumberGenerator_Hybrid::SetSeedTaus2(unsigned int seed){
	m_seedTaus2 = seed;
}

__device__ __host__ unsigned int RandomNumberGenerator_Hybrid::GetSeedTaus3(){
	return m_seedTaus3;
}

__device__ __host__ void RandomNumberGenerator_Hybrid::SetSeedTaus3(unsigned int seed){
	m_seedTaus3 = seed;
}

// Private internal unsigned int generators
__device__ __host__ unsigned int RandomNumberGenerator_Hybrid::LCGStep(){
	unsigned int z = this->GetSeedLCGS();
	unsigned int A = 1664525;
	unsigned int C = 1013904223UL;
	this->SetSeedLCGS(A*z + C);
	return this->GetSeedLCGS();	
}
__device__ __host__ unsigned int RandomNumberGenerator_Hybrid::TausStep1(){
	int S1 = 13, S2 = 19, S3 = 12;
	unsigned int M = 4294967294UL;
	unsigned int z = this->GetSeedTaus1();
	unsigned int b = (((z << S1) ^ z) >> S2);
	this->SetSeedTaus1(((z & M) << S3) ^ b);
	return this->GetSeedTaus1();
}

__device__ __host__ unsigned int RandomNumberGenerator_Hybrid::TausStep2(){
	int S1 = 2, S2 = 25, S3 = 4;
	unsigned int M = 4294967288UL;
	unsigned int z = this->GetSeedTaus2();
	unsigned b = (((z << S1) ^ z) >> S2);
	this->SetSeedTaus2(((z & M) << S3) ^ b);
	return this->GetSeedTaus2();
}

__device__ __host__ unsigned int RandomNumberGenerator_Hybrid::TausStep3(){
	int S1 = 3, S2 = 11, S3 = 17;
	unsigned int M = 4294967280UL;
	unsigned int z = this->GetSeedTaus3();	
	unsigned b = (((z << S1) ^ z) >> S2);
	this->SetSeedTaus3(((z & M) << S3) ^ b);
	return this->GetSeedTaus3();
}

__device__ __host__ double RandomNumberGenerator_Hybrid::HybridGenerator(){
	return 2.3283064365387e-10 * (
		this->TausStep1() ^
		this->TausStep2() ^
		this->TausStep3() ^
		this->LCGStep()
	);
}

// Public methods to generate random numbers
__device__ __host__ double RandomNumberGenerator_Hybrid::GetUniform(){
	double r = this->HybridGenerator();
	return r;
}

__device__ __host__ double RandomNumberGenerator_Hybrid::GetGauss(){
	double u = this->HybridGenerator();
	double v = this->HybridGenerator();
	return sqrt(-2.*log(u)) * cos(2.*M_PI*v);
}

// Public seed manipulation methods
__device__ __host__ void RandomNumberGenerator_Hybrid::ResetSeed(){
	this->SetSeedLCGS(static_cast<unsigned int>(0));
	this->SetSeedTaus1(static_cast<unsigned int>(129));
	this->SetSeedTaus2(static_cast<unsigned int>(130));
	this->SetSeedTaus3(static_cast<unsigned int>(131));
}

__device__ __host__ void RandomNumberGenerator_Hybrid::SetInternalState(unsigned int seedLGCS, unsigned int seedTaus1, unsigned int seedTaus2, unsigned int seedTaus3){
	this->SetSeedLCGS(seedLGCS);
	this->SetSeedTaus1(seedTaus1);
	this->SetSeedTaus2(seedTaus2);
	this->SetSeedTaus3(seedTaus3);	
}
