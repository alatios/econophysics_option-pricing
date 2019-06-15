#include <iostream>
#include <cstdlib>
#include <ctime>		// time(NULL) for seed
#include <random>		// C++11 Mersenne twister
#include <climits>		// UINT_MAX
#include <cmath>		// log, cos, sin, ceil, M_PI
#include <algorithm>	// min
#include <fstream>
#include <cstdio>
#include <vector>
#include "rng.cuh"

using namespace std;

#define NTOGEN 200000	// How many pseudorandom numbers do you need?

__global__ void RNGen_Global(RNGCombinedGenerator *rng_array, unsigned int N_rng, double *res_array, double *gauss_array);
__host__ __device__ void RNGen_HostDev(RNGCombinedGenerator *rng_array, unsigned int N_rng, double *res_array, double *gauss_array, unsigned int tid);
__host__ void RNGen_Host(RNGCombinedGenerator *rng_array, unsigned int N_rng, double *res_array, double *gauss_array);

int main(){

/********************************************************* 	
	How many numbers will each thread generate?
	 static_cast because I never trust int division
	unsigned int rng_num = ceil(static_cast<double>(NTOGEN) / NTHREADMAX);
**********************************************************/

	// Mersenne random generator of unsigned ints, courtesy of C++11
	mt19937 mersennegen(time(NULL));
	uniform_int_distribution<unsigned int> dis(0, UINT_MAX - 128);

	RNGCombinedGenerator *gens = new RNGCombinedGenerator[NTOGEN];
	
	RNG_Generator *generic_rng_pointer = &mersennegen;
	for(unsigned int i=0; i<NTHREADMAX; ++i)
		gens[i].Set_internal_state(generic_rng_pointer);

	double *results = new double[NTOGEN];
	double *gauss_results = new double[NTOGEN];

/*
	////////////// HOST-SIDE GENERATOR //////////////
	RNGen_Host(gens, rng_num, results, gauss_results);
	/////////////////////////////////////////////////
*/


///*
	////////////// DEVICE-SIDE GENERATOR //////////////
	RNGCombinedGenerator *dev_gens;
	double *dev_results, *dev_gauss_results;
	
	cudaMalloc( (void **)&dev_gens, NTHREADMAX*sizeof(RNGCombinedGenerator) );
	cudaMalloc( (void **)&dev_results, NTOGEN*sizeof(double) );
	cudaMalloc( (void **)&dev_gauss_results, NTOGEN*sizeof(double) );
	
	cudaMemcpy(dev_gens, gens, NTHREADMAX*sizeof(RNGCombinedGenerator), cudaMemcpyHostToDevice);
	
	RNGen_Global<<<14,1024>>>(dev_gens, rng_num, dev_results, dev_gauss_results);
	
	cudaMemcpy(results, dev_results, NTOGEN*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(gauss_results, dev_gauss_results, NTOGEN*sizeof(double), cudaMemcpyDeviceToHost);

	cudaFree(dev_gens);
	cudaFree(dev_results);
	cudaFree(dev_gauss_results);
	///////////////////////////////////////////////////
//*/


	cout << "i uniform gauss" << endl;	// For astropy.ascii compatibility
//	for(int i=0; i<NTOGEN; ++i){
	for(unsigned int i=0; i<10; ++i){
		cout << i << " " << results[i] << " " << gauss_results[i] << endl;
	}

	delete[] gens;
	delete[] results;
	delete[] gauss_results;

	return 0;

}

/////////////////////////////////////////////
///////////////// FUNCTIONS /////////////////
/////////////////////////////////////////////

__global__ void RNGen_Global(RNGCombinedGenerator *rng_array, unsigned int N_rng, double *res_array, double *gauss_array){
	unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
	RNGen_HostDev(rng_array, N_rng, res_array, gauss_array, tid);
}

__host__ __device__ void RNGen_HostDev(RNGCombinedGenerator *rng_array, unsigned int N_rng, double *res_array, double *gauss_array, unsigned int tid){
//	double rand_uni, rand_gauss;

	for(unsigned int threadRNG=0; threadRNG<N_rng; ++threadRNG){
		if(N_rng*tid+threadRNG < NTOGEN){
			// Ho verificato che il problema sta nell'implementazione di HybridTaus o dei suoi sottoposti
			res_array[N_rng*tid+threadRNG] = rng_array[tid].GetUniform();
			gauss_array[N_rng*tid+threadRNG] = rng_array[tid].GetGauss();

	// DEBUGGING OUTPUT; REMOVE __device__ AND C0OMMENT __global__ FUNCTION TO USE IT (ONLY WORKS ON CPU)
//			if(N_rng*tid+threadRNG < 10)
//				cout << N_rng*tid+threadRNG << " " << res_array[N_rng*tid+threadRNG] << " " << gauss_array[N_rng*tid+threadRNG] << endl;
		}
	}
}

__host__ void RNGen_Host(RNGCombinedGenerator *rng_array, unsigned int N_rng, double *res_array, double *gauss_array){
	for(unsigned int tid=0; tid<NTHREADMAX; ++tid)
		RNGen_HostDev(rng_array, N_rng, res_array, gauss_array, tid);
}
