//
//	INTER- AND INTRA-STREAM CORRELATION TEST
//
//	Tests correlation of random generated numbers within single streams
//	as well as across all streams.
//

#include <iostream>
#include <ctime>		// time(NULL) for seed
#include <climits>		// UINT_MAX

#include "RNG.cuh"
#include "CorrelationToolbox.cuh"

using namespace std;

int main(){
	
	unsigned int numberOfBlocks = 1;
	unsigned int numberOfThreadsPerBlock = 512;
	unsigned int totalNumberOfThreads = numberOfBlocks * numberOfThreadsPerBlock;
	unsigned int numbersToGeneratePerThread = 1000;
	unsigned int totalNumbersToGenerate = totalNumberOfThreads * numbersToGeneratePerThread;

	cout << "Total numbers to generate: " << totalNumbersToGenerate << endl;
	cout << "Total number of threads: " << totalNumberOfThreads << endl;
	cout << "Total numbers to generate per thread: " << numbersToGeneratePerThread << endl;
	
	double **uniformNumbers = new double*[totalNumbersToGenerate];
	double **gaussianNumbers = new double*[totalNumbersToGenerate];
	
	for(unsigned int threadIndex=0; threadIndex<totalNumberOfThreads; ++threadIndex){
		uniformNumbers[threadIndex] = new double[numbersToGeneratePerThread];
		gaussianNumbers[threadIndex] = new double[numbersToGeneratePerThread];
	}

	// Seed for tausworthe support generator
	unsigned int seed;	
	do
		seed = time(NULL);
	while(seed < 129 || seed > UINT_MAX - totalNumberOfThreads);
	

	RandomNumberGeneration(numberOfBlocks, numberOfThreadsPerBlock, uniformNumbers, gaussianNumbers, totalNumbersToGenerate, numbersToGeneratePerThread, seed);
	
	cout << endl <<  "############### INTRA-STREAM TEST ###############" << endl;
	
	// Average, exp. 0.5 for uniform numbers, 0 for gaussian numbers
	double *uniformAverages = new double[totalNumberOfThreads];
	double *gaussAverages = new double[totalNumberOfThreads];
	
	cout << endl << "Uniform averages (exp. 0.5):" << endl;
	EvaluateStreamAverage(uniformNumbers, uniformAverages, totalNumberOfThreads, numbersToGeneratePerThread, false);
	cout << endl << "Gaussian averages (exp. 0):" << endl;
	EvaluateStreamAverage(gaussianNumbers, gaussAverages, totalNumberOfThreads, numbersToGeneratePerThread, false);

	// Gaussian variance, exp. 1
	double *gaussVariances = new double[totalNumberOfThreads];
	cout << endl << "Gaussian variances (exp. 1):" << endl;
	EvaluateStreamVariance(gaussianNumbers, gaussAverages, gaussVariances, totalNumberOfThreads, numbersToGeneratePerThread, false);
	
	// Gaussian kurtosis, exp. 3
	double *gaussKurtosises = new double[totalNumberOfThreads];
	cout << endl << "Gaussian kurtosises (exp. 3):" << endl;
	EvaluateStreamKurtosis(gaussianNumbers, gaussAverages, gaussVariances, gaussKurtosises, totalNumberOfThreads, numbersToGeneratePerThread, false);

	// Autocorrelation i/i+1, exp. 0.25 for uniform numbers, 0 for gaussian numbers
	double *uniformCorrelations = new double[totalNumberOfThreads];
	double *gaussCorrelations = new double[totalNumberOfThreads];

	cout << endl << "Uniform autocorrelations i/i+k (exp. 0.25):" << endl;
	EvaluateCompleteAutocorrelation(uniformNumbers, uniformCorrelations, totalNumberOfThreads, numbersToGeneratePerThread, true);
	cout << endl << "Gaussian autocorrelations i/i+k (exp. 0):" << endl;
	EvaluateCompleteAutocorrelation(gaussianNumbers, gaussCorrelations, totalNumberOfThreads, numbersToGeneratePerThread, true);

	
/** Per riferimento, i test precedenti
 
	cout << endl <<  "############### INTER-STREAM TEST ###############" << endl << endl;
	int streamStep = 0;					// To choose which step of the random generation you want
	double sumOfUniformNumbers = 0.;
	double squaredSumOfUniformNumbers = 0.;
	double sumOfGaussianNumbers = 0.;
	double squaredSumOfGaussianNumbers = 0.;

	for(unsigned int randomStream=0; randomStream<totalNumberOfThreads - 1; ++randomStream){
		sumOfUniformNumbers += uniformNumbers[randomStream * numbersToGeneratePerThread + streamStep]
								* uniformNumbers[(randomStream + 1) * numbersToGeneratePerThread + streamStep];
		sumOfGaussianNumbers += gaussianNumbers[randomStream * numbersToGeneratePerThread + streamStep]
								* gaussianNumbers[(randomStream + 1) * numbersToGeneratePerThread + streamStep];
		squaredSumOfUniformNumbers += pow(uniformNumbers[randomStream * numbersToGeneratePerThread + streamStep],2)
								* pow(uniformNumbers[(randomStream + 1) * numbersToGeneratePerThread + streamStep],2);
		squaredSumOfGaussianNumbers += pow(gaussianNumbers[randomStream * numbersToGeneratePerThread + streamStep],2)
								* pow(gaussianNumbers[(randomStream + 1) * numbersToGeneratePerThread + streamStep],2);
	}

	double averageOfUniformNumbers = sumOfUniformNumbers/totalNumberOfThreads;
	double standardDeviationOfUniformNumbers = sqrt(squaredSumOfUniformNumbers/totalNumberOfThreads - pow(averageOfUniformNumbers,2));
	double averageOfGaussianNumbers = sumOfGaussianNumbers/totalNumberOfThreads;
	double standardDeviationOfGaussianNumbers = sqrt(squaredSumOfGaussianNumbers/totalNumberOfThreads - pow(averageOfGaussianNumbers,2));
	cout << "Correlation function of uniform numbers: " << averageOfUniformNumbers << " +- " << standardDeviationOfUniformNumbers << endl;
	cout << "The value obtained differs from the expected by: " << abs(averageOfUniformNumbers/standardDeviationOfUniformNumbers) << " stardard deviation" << endl << endl;
	cout << "Correlation function of gaussian numbers: " << averageOfGaussianNumbers << " +- " << standardDeviationOfGaussianNumbers << endl;
	cout << "The value obtained differs from the expected by: " << abs(averageOfGaussianNumbers/standardDeviationOfGaussianNumbers) << " stardard deviation" << endl << endl;

	cout << endl <<  "############### INTRA-STREAM TEST ###############" << endl << endl;

	int threadToTest = 0;			// To choose on which thread make tests
	cout << "The tests is made only on thread stream no. "  << threadToTest << endl;
	sumOfUniformNumbers = 0;
	squaredSumOfUniformNumbers = 0;
	double sumOfUniformNumberProducts = 0.;
	double sumOfGaussianNumberProducts = 0.;
	double squaredSumOfUniformNumberProducts = 0.;
	double squaredSumOfGaussianNumberProducts = 0.;
	sumOfGaussianNumbers = 0;
	squaredSumOfGaussianNumbers = 0;
	for(unsigned int randomNumber=0; randomNumber<numbersToGeneratePerThread; ++randomNumber){
		sumOfUniformNumbers += uniformNumbers[randomNumber + threadToTest * numbersToGeneratePerThread]; 
		sumOfGaussianNumbers += gaussianNumbers[randomNumber + threadToTest * numbersToGeneratePerThread];
		squaredSumOfGaussianNumbers += pow(gaussianNumbers[randomNumber + threadToTest * numbersToGeneratePerThread],2);
		squaredSumOfUniformNumbers += pow(uniformNumbers[randomNumber + threadToTest * numbersToGeneratePerThread], 2);
	}
	
	for(unsigned int randomNumber=0; randomNumber<numbersToGeneratePerThread - 1; ++randomNumber){
		sumOfUniformNumberProducts += uniformNumbers[randomNumber + threadToTest * numbersToGeneratePerThread] * uniformNumbers[randomNumber + 1 + threadToTest * numbersToGeneratePerThread];
		sumOfGaussianNumberProducts += gaussianNumbers[randomNumber + threadToTest * numbersToGeneratePerThread] * gaussianNumbers[randomNumber + 1 + threadToTest * numbersToGeneratePerThread];
		squaredSumOfUniformNumberProducts += pow(uniformNumbers[randomNumber + threadToTest * numbersToGeneratePerThread], 2) * pow(uniformNumbers[randomNumber + 1 + threadToTest * numbersToGeneratePerThread], 2);
		squaredSumOfGaussianNumberProducts += pow(gaussianNumbers[randomNumber + threadToTest * numbersToGeneratePerThread], 2) * pow(gaussianNumbers[randomNumber + 1 + threadToTest * numbersToGeneratePerThread], 2);
	}

	averageOfUniformNumbers = sumOfUniformNumbers/numbersToGeneratePerThread;
	standardDeviationOfUniformNumbers = sqrt(squaredSumOfUniformNumbers/numbersToGeneratePerThread - pow(averageOfUniformNumbers,2));
	double correlationOfUniformNumbers = sumOfUniformNumberProducts/numbersToGeneratePerThread;
	averageOfGaussianNumbers = sumOfGaussianNumbers/numbersToGeneratePerThread;
	standardDeviationOfGaussianNumbers = sqrt(squaredSumOfGaussianNumbers/numbersToGeneratePerThread - pow(averageOfGaussianNumbers,2));
	cout << "Uniform mean: " << averageOfUniformNumbers << "| Uniform standard deviation: " << standardDeviationOfUniformNumbers << "| Uniform Correlation intrastram: " << correlationOfUniformNumbers << endl;
	cout << "Gaussian mean: " << averageOfGaussianNumbers << "| Gaussian standard deviation: " << standardDeviationOfGaussianNumbers << endl << endl;

//Non so che formula abbia usato Riccardo qui sopra e con che criterio il test sul singolo thread sia stato fatto diverso
//da quello su tutti i thread quindi ho riportato il procedimento precedente anche in questo test intrastream perchè la 
//correlazione tra i numeri casuali sul singolo thread e tra quelli i-esimi di ogni thread dovrebbe essere la stessa credo
//Se pensate che sia sbagliato ho lasciato inalterati i cout sopra, così al massimo basta cancellare le righe in eccesso

	cout << "Correlation function of uniform numbers: " << averageOfUniformNumbers << " +- " << standardDeviationOfUniformNumbers << endl;
	cout << "The value obtained differs from the expected by: " << abs(averageOfUniformNumbers/standardDeviationOfUniformNumbers) << " stardard deviation" << endl << endl;
	cout << "Correlation function of gaussian numbers: " << averageOfGaussianNumbers << " +- " << standardDeviationOfGaussianNumbers << endl;
	cout << "The value obtained differs from the expected by: " << abs(averageOfGaussianNumbers/standardDeviationOfGaussianNumbers) << " stardard deviation" << endl << endl;

*/

	delete[] uniformAverages;
	delete[] gaussAverages;
	delete[] gaussVariances;
	delete[] gaussKurtosises;
	delete[] uniformCorrelations;
	delete[] gaussCorrelations;

	for(unsigned int threadIndex=0; threadIndex<totalNumberOfThreads; ++threadIndex){
		delete[] uniformNumbers[threadIndex];
		delete[] gaussianNumbers[threadIndex];
	}

	delete[] uniformNumbers;
	delete[] gaussianNumbers;

	return 0;

}
