#include "rng.cuh"

using namespace std;

//LCGS

__device__ __host__ unsigned RNG_LCGS::GetSeed() const{
	return _seed;
}

__device__ __host__ void RNG_LCGS::SetSeed(unsigned int seed){
	_seed = seed;
}


__device__ __host__ RNG_LCGS::RNG_LCGS(){
	this->SetSeed(static_cast<unsigned int>(0));
}

__device__ __host__ RNG_LCGS::RNG_LCGS(const RNG_LCGS& lcgs){
	this->SetSeed(lcgs.GetSeed());
}

__device__ __host__ RNG_LCGS::RNG_LCGS(unsigned int seed){
	this.SetSeed(seed);
}

__device__ __host__ void RNG_LCGS::ResetSeed(){
	this->SetSeed(static_cast<unsigned int>(0));
}

__device__ __host__ double RNG_LCGS::GetUniform(){
	unsigned int z = this->GetSeed();
	this->SetSeed(_A*z + _C);
	
	return this->GetSeed();
}

__device__ __host__ double RNG_LCGS::GetGauss(){
	unsigned int u = this->GetUniform();
	unsigned int v = this->GetUniform();

	return sqrt(-2.*log(u)) * cos(2.*M_PI*v);
}

__device__ __host__ RNG_LCGS& RNG_LCGS::Set_internal_state(unsigned int * seed){
	this->SetSeed(*seed);
	return *this;
}

//Tausworthe

__device__ __host__ unsigned int RNG_Tausworthe::GetM() const{
	return _M;
}
	
__device__ __host__ void RNG_Tausworthe::SetM(unsigned int m){
	_M = m;
}
	
__device__ __host__ int RNG_Tausworthe::GetK1() const{
	return _k1;
}

__device__ __host__ void RNG_Tausworthe::SetK1(int k1){
	_k1 = k1;
}

__device__ __host__ int RNG_Tausworthe::GetK2() const{
	return k2;
}
	
__device__ __host__ void RNG_Tausworthe::SetK2(int k2){
	_k2 = k2;
}

__device__ __host__ int RNG_Tausworthe::GetK3() const{
	return k3;
}

__device__ __host__ void RNG_Tausworthe::SetK3(int k3){
	_k3 = k3;
}

__device__ __host__ unsigned RNG_Tausworthe::GetSeed() const{
	return _seed;
}

__device__ __host__ void RNG_Tausworthe::SetSeed(unsigned int seed){
	_seed = 128 + seed;
}

__device__ __host__ RNG_Tausworthe::RNG_Tausworthe(){
	this->SetSeed(129);
	this->SetM(4294967288UL);
	this->SetK1(13);
	this->SetK2(19);
	this->SetK3(12);
}

__device__ __host__ RNG_Tausworthe::RNG_Tausworthe(const RNG_Tausworthe& taus){
	this->SetSeed(taus.GetSeed());
	this->SetM(taus.GetM());
	this->SetK1(taus.GetK1());
	this->SetK2(taus.GetK2());
	this->SetK3(taus.GetK3());
}

__device__ __host__ RNG_Tausworthe::RNG_Tausworthe(unsigned int seed, unsigned int M, int k1, int k2, int k3){
	this->SetSeed(seed);
	this->SetM(M);
	this->SetK1(k1);
	this->SetK2(k2);
	this->SetK3(k3);
}

__device__ __host__ void RNG_Tausworthe::ResetSeed(){
	this->SetSeed(static_cast<unsigned int>(129));

}

__device__ __host__ virtual double RNG_Tausworthe::GetUniform(){
	unsigned int z = this->GetSeed();
	unsigned b = (((z << this->GetK1()) ^ z) >> this->GetK2());
	
	this->SetSeed(((z & this->GetM()) << this->GetK3()) ^ b);
	return this->GetSeed();
}

__device__ __host__ virtual double RNG_Tausworthe::GetGauss(){
	unsigned int u = this->GetUniform();
	unsigned int v = this->GetUniform();

	return sqrt(-2.*log(u)) * cos(2.*M_PI*v);
}

__device__ __host__ RNG_Tausworthe& RNG_Tausworthe::Set_internal_state(unsigned int * seed){
	this->SetSeed(*seed);
	return *this;
}

//Combined generator

__device__ __host__ RNG_Tausworthe& RNGCombinedGenerator::GetTausComponent(int component) const{
	return _taus[component];
}

__device__ __host__	void RNGCombinedGenerator::SetTausComponent(int component, const RNG_Tausworthe& rng_taus){
	_taus[component] = rng_taus;
}

__device__ __host__	RNG_LCGS& RNGCombinedGenerator::GetLcgs() const{
	return _lcgs;
}

__device__ __host__	void RNGCombinedGenerator::SetLcgs(const RNG_LCGS& rng_lcgs){
	_lcgs = rng_lcgs;
}

__device__ __host__ RNGCombinedGenerator::RNGCombinedGenerator(){
	for(int i=0; i<3; ++i)
		this->SetTausComponent(i, RNG_Tausworthe());
	this->SetLcgs(RNG_LCGS());
}

__device__ __host__ RNGCombinedGenerator::RNGCombinedGenerator(const RNGCombinedGenerator& gen2){
	this->SetLcgs(gen2.GetLcgs());
	for(int i=0; i<3; i++)
		this->SetTausComponent(gen2.GetTausComponent());
}

__device__ __host__ unsigned RNGCombinedGenerator::TausStep1(){
	this->GetTausComponent(1).SetK1(13);
	this->GetTausComponent(1).SetK2(19);
	this->GetTausComponent(1).SetK3(12);
	this->GetTausComponent(1).SetM(4294967294UL);

	unsigned int z = this->GetLcgs().GetSeed();

	unsigned b = (((z << this->GetTausComponent(1).GetK1()) ^ z) >> this->GetTausComponent(1).GetK2());
	
	this->GetLcgs().SetSeed(((z & this->GetTausComponent(1).GetM()) << this->GetTausComponent(1).GetK3()) ^ b);
	return this->GetLcgs().GetSeed();
}

__device__ __host__ unsigned RNGCombinedGenerator::TausStep2(){
	this->GetTausComponent(2).SetK1(2);
	this->GetTausComponent(2).SetK2(25);
	this->GetTausComponent(2).SetK3(4);
	this->GetTausComponent(2).SetM(4294967288UL);

	unsigned int z = this->GetTausComponent(1).GetSeed();

	unsigned b = (((z << this->GetTausComponent(2).GetK1()) ^ z) >> this->GetTausComponent(2).GetK2());
	
	this->GetTausComponent(1).SetSeed(((z & this->GetTausComponent(2).GetM()) << this->GetTausComponent(2).GetK3()) ^ b);
	return this->GetTausComponent(1).GetSeed();
}

__device__ __host__ unsigned RNGCombinedGenerator::TausStep3(){
	this->GetTausComponent(3).SetK1(3);
	this->GetTausComponent(3).SetK2(11);
	this->GetTausComponent(3).SetK3(17);
	this->GetTausComponent(3).SetM(4294967280UL);

	unsigned int z = this->GetTausComponent(2).GetSeed();

	unsigned b = (((z << this->GetTausComponent(3).GetK1()) ^ z) >> this->GetTausComponent(3).GetK2());
	
	this->GetTausComponent(2).SetSeed(((z & this->GetTausComponent(3).GetM()) << this->GetTausComponent(3).GetK3()) ^ b);
	return this->GetTausComponent(2).GetSeed();
}

__device__ __host__ unsigned RNGCombinedGenerator::LCGStep(){
	unsigned int z = this->GetTausComponent(3).GetSeed();

	this->GetTausComponent(3).SetSeed((this->GetLcgs()._A)*z + this->GetLcgs()._C);
	return this->GetTausComponent(3).GetSeed();
}

__device__ __host__ double RNGCombinedGenerator::HybridTausGenerator(){
	return 2.3283064365387e-10 * (
		this->TausStep1() ^
		this->TausStep2() ^
		this->TausStep3() ^
		this->LCGStep()
	);
}

__device__ __host__ void RNGCombinedGenerator::ResetSeed(){
	this->GetLcgs().ResetSeed();
	this->GetTausComponent(1).ResetSeed();
	this->GetTausComponent(2).ResetSeed();
	this->GetTausComponentT(3).ResetSeed();
}

__device__ __host__ double RNGCombinedGenerator::GetUniform(){
	double r = this->HybridTausGenerator();
	return r;
//	return 0.25;
}

__device__ __host__ double RNGCombinedGenerator::GetGauss(){
	double u = this->HybridTausGenerator();
	double v = this->HybridTausGenerator();
//	double v = 0.2;

	return sqrt(-2.*log(u)) * cos(2.*M_PI*v);
}

__device__ __host__ RNGCombinedGenerator& RNGCombinedGenerator::Set_internal_state(unsigned int * seed){
	this->GetLcgs().SetSeed(seed[0]);
	for(int i=1; i<4; i++)
		this->GetTausComponent(i).SetSeed(seed[i]);
	return *this;
}

