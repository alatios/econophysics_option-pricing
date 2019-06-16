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

int main(){
	
	unsigned int numberOfBlocks = 2;
	unsigned int numberOfThreadsPerBlock = 512;
	unsigned int totalNumberOfThreads = numberOfBlocks * numberOfThreadsPerBlock;
	
	unsigned int totalNumbersToGenerate = 10;
	unsigned int numbersToGeneratePerThread = ceil(static_cast<double>(totalNumbersToGenerate) / totalNumberOfThreads);
	
	mt19937 mersenneCoreGenerator(time(NULL));
	uniform_int_distribution<unsigned int> mersenneDistribution(129, UINT_MAX);
	
	cout << "Numbers to generate: " << totalNumbersToGenerate << endl;
	cout << "Total number of threads: " << totalNumberOfThreads << endl;
	cout << "Numbers each thread generates (round up): " << numbersToGeneratePerThread << endl;
	
	RandomNumberGenerator* generator = new RandomNumberGenerator_Hybrid(mersenneDistribution(mersenneCoreGenerator), mersenneDistribution(mersenneCoreGenerator), mersenneDistribution(mersenneCoreGenerator), mersenneDistribution(mersenneCoreGenerator));

	for(int i=0; i<totalNumbersToGenerate; ++i)
		cout << generator->GetUniform() << " " << generator->GetGauss() << endl;
		
	delete generator;

	return 0;
}
