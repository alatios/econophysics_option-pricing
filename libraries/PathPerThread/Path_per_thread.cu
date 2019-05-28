#include "Path_per_thread.cuh"

using namespace std;

//Constructor
__device__ __host__ Path_per_thread::Path_per_thread(){
	this->SetNumberOfPathsPerThread(5);
	this->SetIsCorrectDimension(true);
	
	_PathsPerThread = new Path[5];
}

__device__ __host__ Path_per_thread::Path_per_thread(unsigned int NumberOfPathsPerThread){
	this->SetIsCorrectDimension(true);
	this->SetNumberOfPathsPerThread(NumberOfPathsPerThread);
	_PathsPerThread = new Path[NumberOfPathsPerThread];
}


__device__ __host__ Path_per_thread::Path_per_thread(unsigned int NumberOfPathsPerThread, Path* PathArray){
	this->SetIsCorrectDimension(true);
	this->SetNumberOfPathsPerThread(NumberOfPathsPerThread);
	_PathsPerThread = new Path[NumberOfPathsPerThread];

	for(unsigned int i=0; i<NumberOfPathsPerThread; ++i)
		this->SetPathComponent(i, PathArray[i]);
}

__device__ __host__ Path_per_thread::~Path_per_thread(){
	delete[] _PathsPerThread;
}

//Methods
__device__ __host__ Path& Path_per_thread::GetPathComponent(unsigned int NumberOfPath){
	if(NumberOfPath > this->GetNumberOfPathsPerThread())
		this->SetIsCorrectDimension(false);
	
	return _PathsPerThread[NumberOfPath];
}

__device__ __host__ void Path_per_thread::SetPathComponent(unsigned int NumberOfPath, Path SinglePath){
	if(NumberOfPath > this->GetNumberOfPathsPerThread())
		this->SetIsCorrectDimension(false);

	_PathsPerThread[NumberOfPath].SetGaussianRandomVariable(SinglePath.GetGaussianRandomVariable());
	_PathsPerThread[NumberOfPath].SetSpotPrice(SinglePath.GetSpotPrice());
	_PathsPerThread[NumberOfPath].SetInputMarketData(SinglePath.GetInputMarketData());
	_PathsPerThread[NumberOfPath].SetInputOptionData(SinglePath.GetInputOptionData());	
}

__device__ __host__ unsigned int Path_per_thread::GetNumberOfPathsPerThread(){
	return _NumberOfPathsPerThread;
}

__device__ __host__ void Path_per_thread::SetNumberOfPathsPerThread(unsigned int NumberOfPathsPerThread){
	_NumberOfPathsPerThread = NumberOfPathsPerThread;
}

__device__ __host__ void Path_per_thread::SetIsCorrectDimension(bool IsCorrect){
	_IsCorrectDimension = IsCorrect;
}

__device__ __host__ bool Path_per_thread::GetIsCorrectDimension(){
	return _IsCorrectDimension;
}

__device__ __host__ void Path_per_thread::SetPathArray(unsigned int ArrayDimension, Path* PathArray){
	if(ArrayDimension != this->GetNumberOfPathsPerThread())
		this->SetIsCorrectDimension(false);
	
	for(unsigned int i=0; i<this->GetNumberOfPathsPerThread(); ++i)
		this->SetPathComponent(i, PathArray[i]);
}	
