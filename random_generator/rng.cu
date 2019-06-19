#include "rng.cuh"

using namespace std;

__device__ __host__ unsigned int RNGCombinedGenerator::GetSeedLCGS() const{
	return _seedLCGS;
}

__device__ __host__ void RNGCombinedGenerator::SetSeedLCGS(const unsigned int seed){
	_seedLCGS = seed;
}

__device__ __host__ unsigned int RNGCombinedGenerator::GetSeedTaus1() const{
	return _seedTaus1;
}

__device__ __host__ void RNGCombinedGenerator::SetSeedTaus1(const unsigned int seed){
	_seedTaus1 = seed;
}

__device__ __host__ unsigned int RNGCombinedGenerator::GetSeedTaus2() const{
	return _seedTaus2;
}

__device__ __host__ void RNGCombinedGenerator::SetSeedTaus2(const unsigned int seed){
	_seedTaus2 = seed;
}

__device__ __host__ unsigned int RNGCombinedGenerator::GetSeedTaus3() const{
	return _seedTaus3;
}

__device__ __host__ void RNGCombinedGenerator::SetSeedTaus3(const unsigned int seed){
	_seedTaus3 = seed;
}

__device__ __host__ RNGCombinedGenerator::RNGCombinedGenerator(const unsigned int seed1, const unsigned int seed2, const unsigned int seed3, const unsigned int seed4){
	this->SetSeedLCGS(seed1);
	this->SetSeedTaus1(seed2);
	this->SetSeedTaus2(seed3);
	this->SetSeedTaus3(seed4);
}

__device__ __host__ RNGCombinedGenerator::RNGCombinedGenerator(){
	this->SetSeedLCGS(static_cast<unsigned int>(0));
	this->SetSeedTaus1(static_cast<unsigned int>(129));
	this->SetSeedTaus2(static_cast<unsigned int>(130));
	this->SetSeedTaus3(static_cast<unsigned int>(131));
}

__device__ __host__ RNGCombinedGenerator::RNGCombinedGenerator(const RNGCombinedGenerator& gen2){
	this->SetSeedLCGS(gen2.GetSeedLCGS());
	this->SetSeedTaus1(gen2.GetSeedTaus1());
	this->SetSeedTaus2(gen2.GetSeedTaus2());
	this->SetSeedTaus3(gen2.GetSeedTaus3());
}

__device__ __host__ unsigned int RNGCombinedGenerator::TausStep1(){
	int S1 = 13, S2 = 19, S3 = 12;
	unsigned int M = 4294967294UL;
	unsigned int z = this->GetSeedTaus1();
	
	unsigned b = (((z << S1) ^ z) >> S2);
	
	this->SetSeedTaus1(((z & M) << S3) ^ b);
	return this->GetSeedTaus1();
}

__device__ __host__ unsigned int RNGCombinedGenerator::TausStep2(){
	int S1 = 2, S2 = 25, S3 = 4;
	unsigned int M = 4294967288UL;
	unsigned int z = this->GetSeedTaus2();
	
	unsigned b = (((z << S1) ^ z) >> S2);

	this->SetSeedTaus2(((z & M) << S3) ^ b);
	return this->GetSeedTaus2();
}
__device__ __host__ unsigned int RNGCombinedGenerator::TausStep3(){
	int S1 = 3, S2 = 11, S3 = 17;
	unsigned int M = 4294967280UL;
	unsigned int z = this->GetSeedTaus3();
	
	unsigned b = (((z << S1) ^ z) >> S2);

	this->SetSeedTaus3(((z & M) << S3) ^ b);
	return this->GetSeedTaus3();
}

__device__ __host__ unsigned int RNGCombinedGenerator::LCGStep(){
	unsigned int z = this->GetSeedLCGS();
	unsigned int A = 1664525;
	unsigned int C = 1013904223UL;
	this->SetSeedLCGS(A*z + C);
	return this->GetSeedLCGS();
}

__device__ __host__ unsigned int RNGCombinedGenerator::HybridTausGenerator(){
	return (
		this->TausStep1() ^
		this->TausStep2() ^
		this->TausStep3() ^
		this->LCGStep()
	);
}

__device__ __host__ void RNGCombinedGenerator::ResetSeed(){
	this->SetSeedLCGS(static_cast<unsigned int>(0));
	this->SetSeedTaus1(static_cast<unsigned int>(1));
	this->SetSeedTaus2(static_cast<unsigned int>(2));
	this->SetSeedTaus3(static_cast<unsigned int>(3));
}

__device__ __host__ double RNGCombinedGenerator::GetUnsignedInt(){
	return this->HybridTausGenerator();
}

__device__ __host__ double RNGCombinedGenerator::GetUniform(){
	return 2.3283064365387e-10 * this->GetUnsignedInt();
}

__device__ __host__ double RNGCombinedGenerator::GetGauss(){
	double u = this->GetUniform();
	double v = this->GetUniform();

	return sqrt(-2.*log(u)) * cos(2.*M_PI*v);
}

__device__ __host__ void RNGCombinedGenerator::SetInternalState(unsigned int seedLGCS, unsigned int seedTaus1, unsigned int seedTaus2, unsigned int seedTaus3){
	this->SetSeedLCGS(seedLGCS);
	this->SetSeedTaus1(seedTaus1);
	this->SetSeedTaus2(seedTaus2);
	this->SetSeedTaus3(seedTaus3);
}
