#include <iostream>
#include <cmath>		// sqrt
#include "rng.cuh"
#include "correlationtest.cuh"

using namespace std;

void correlationTest(unsigned int totalNumberOfThreads, unsigned int numbersToGeneratePerThread, double* uniformNumbers, double* gaussianNumbers){

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
}