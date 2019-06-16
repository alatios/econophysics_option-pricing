#include "rng.cuh"

using namespace std;

// MERSENNE

__device__ __host__ RandomNumberGenerator_Mersenne::RandomNumberGenerator_Mersenne(){
	m_CoreMersenneGenerator = new mt19937(0);
}

__device__ __host__ RandomNumberGenerator_Mersenne::RandomNumberGenerator_Mersenne(unsigned int seed){
	m_CoreMersenneGenerator = new mt19937(seed);
}

__device__ __host__ RandomNumberGenerator_Mersenne::~RandomNumberGenerator_Mersenne(){
	delete m_CoreMersenneGenerator;
}

__device__ __host__ unsigned int RandomNumberGenerator_Mersenne::GetUnsignedInt({
	uniform_int_distribution<unsigned int> distribution(0, UINT_MAX);
	return distribution(m_CoreMersenneGenerator);
}
__device__ __host__ double RandomNumberGenerator_Mersenne::GetUniform(){
	uniform_real_distribution<double> distribution(0.,1.);
	return distribution(m_CoreMersenneGenerator);
}
__device__ __host__ double RandomNumberGenerator_Mersenne::GetGauss(){
	normal_distribution<double> distribution{0,1};
	return distribution(m_CoreMersenneGenerator);
}
__device__ __host__ void RandomNumberGenerator_Mersenne::ResetSeed(){
	m_CoreMersenneGenerator.seed(0);
}

__device__ __host__ void RandomNumberGenerator_Mersenne::SetInternalState(RandomNumberGenerator *supportGenerator){
	m_CoreMersenneGenerator.seed(supportGenerator->GetUnsignedInt());
}

__device__ __host__ void RandomNumberGenerator_Mersenne::SetInternalState(unsigned int seed){
		m_CoreMersenneGenerator.seed(seed);
}
