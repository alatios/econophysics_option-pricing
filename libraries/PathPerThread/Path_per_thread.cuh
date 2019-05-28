#ifndef __Path_per_thread_h__
#define __Path_per_thread_h__

#include "../Path/Path.cuh"
#include <iostream>
#include <cmath>

//#define N 5

using namespace std;

class Path_per_thread{

private:

	Path* _PathsPerThread;
	unsigned int _NumberOfPathsPerThread;
	bool _IsCorrectDimension;
	
	__device__ __host__ void SetIsCorrectDimension(bool);
	__device__ __host__ void SetNumberOfPathsPerThread(unsigned int);

public:

	__device__ __host__ Path_per_thread();
	__device__ __host__ Path_per_thread(unsigned int);
	__device__ __host__ Path_per_thread(unsigned int, Path*);
	__device__ __host__ ~Path_per_thread();

	__device__ __host__ void SetPathArray(unsigned int, Path*);
	__device__ __host__ Path& GetPathComponent(unsigned int);
	__device__ __host__ void SetPathComponent(unsigned int, Path);
	
	__device__ __host__ unsigned int GetNumberOfPathsPerThread();
	__device__ __host__	bool GetIsCorrectDimension();
	
};
#endif
