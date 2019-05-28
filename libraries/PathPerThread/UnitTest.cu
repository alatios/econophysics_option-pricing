#include "Path_per_thread.cuh"
#include <iostream>

using namespace std;
/*

OUTPUT:

  No problem ----> 1
  There is a problem ----> 0

*/

int main(){
	
	Path_per_thread ppt1;
	bool* tests = new bool[10];
	bool test_dim;
	
	Input_market_data MarketData(200., 3., 2.);
	Input_option_data OptionData(250., 20, 40., 'p');	// DeltaTime = 2
	
	Path* paths = new Path[10];
	for(unsigned int i=0; i<10; ++i){
		paths[i].SetGaussianRandomVariable(1.);
		paths[i].SetSpotPrice(2.);
		paths[i].SetInputMarketData(MarketData);
		paths[i].SetInputOptionData(OptionData);
	}
	
	Path_per_thread ppt2(10, paths);
	
	Path_per_thread ppt3(7);
	
	cout << endl << "-------------PathPerThread_test-------------" << endl;\
	cout << "Constructors testing" << endl;
	
	test_dim = (ppt1.GetNumberOfPathsPerThread() == static_cast<unsigned int>(5));
	cout << test_dim << endl;
	for(unsigned int i=0; i<ppt1.GetNumberOfPathsPerThread(); ++i){		
		tests[i] = (
			ppt1.GetPathComponent(i).GetGaussianRandomVariable() == static_cast<float>(0.)
			&& ppt1.GetPathComponent(i).GetSpotPrice() == static_cast<float>(0.)
			&& ppt1.GetPathComponent(i).GetInputMarketData().GetZeroPrice()==static_cast<float>(100.)
			&& ppt1.GetPathComponent(i).GetInputMarketData().GetVolatility()==static_cast<float>(0.25)
			&& ppt1.GetPathComponent(i).GetInputMarketData().GetRiskFreeRate()==static_cast<float>(0.1)
			&& ppt1.GetPathComponent(i).GetInputOptionData().GetStrikePrice()==static_cast<float>(110.)
			&& ppt1.GetPathComponent(i).GetInputOptionData().GetNumberOfIntervals()==static_cast<unsigned int>(365)
			&& ppt1.GetPathComponent(i).GetInputOptionData().GetTimeToMaturity()==static_cast<float>(365.)
			&& ppt1.GetPathComponent(i).GetInputOptionData().GetDeltaTime()==static_cast<float>(1.)
			&& ppt1.GetPathComponent(i).GetInputOptionData().GetOptionType()==static_cast<char>('c')
		);
		
		cout << tests[i] << "\t";
	}
	cout << endl;
	cout << ppt1.GetIsCorrectDimension() << endl << endl;
	
	test_dim = (ppt2.GetNumberOfPathsPerThread() == static_cast<unsigned int>(10));
	cout << test_dim << endl;
	for(unsigned int i=0; i<ppt2.GetNumberOfPathsPerThread(); ++i){		
		tests[i] = (
			ppt2.GetPathComponent(i).GetGaussianRandomVariable() == static_cast<float>(1.)
			&& ppt2.GetPathComponent(i).GetSpotPrice() == static_cast<float>(2.)
			&& ppt2.GetPathComponent(i).GetInputMarketData().GetZeroPrice()==static_cast<float>(200.)
			&& ppt2.GetPathComponent(i).GetInputMarketData().GetVolatility()==static_cast<float>(3.)
			&& ppt2.GetPathComponent(i).GetInputMarketData().GetRiskFreeRate()==static_cast<float>(2.)
			&& ppt2.GetPathComponent(i).GetInputOptionData().GetStrikePrice()==static_cast<float>(250.)
			&& ppt2.GetPathComponent(i).GetInputOptionData().GetNumberOfIntervals()==static_cast<unsigned int>(20)
			&& ppt2.GetPathComponent(i).GetInputOptionData().GetTimeToMaturity()==static_cast<float>(40.)
			&& ppt2.GetPathComponent(i).GetInputOptionData().GetDeltaTime()==static_cast<float>(2.)
			&& ppt2.GetPathComponent(i).GetInputOptionData().GetOptionType()==static_cast<char>('p')
		);
		cout << tests[i] << "\t";
	}
	cout << endl;
	cout << ppt2.GetIsCorrectDimension() << endl << endl;
	
	test_dim = (ppt3.GetNumberOfPathsPerThread() == static_cast<unsigned int>(7));
	cout << test_dim << endl;
	for(unsigned int i=0; i<ppt3.GetNumberOfPathsPerThread(); ++i){		
		tests[i] = (
			ppt3.GetPathComponent(i).GetGaussianRandomVariable() == static_cast<float>(0.)
			&& ppt3.GetPathComponent(i).GetSpotPrice() == static_cast<float>(0.)
			&& ppt3.GetPathComponent(i).GetInputMarketData().GetZeroPrice()==static_cast<float>(100.)
			&& ppt3.GetPathComponent(i).GetInputMarketData().GetVolatility()==static_cast<float>(0.25)
			&& ppt3.GetPathComponent(i).GetInputMarketData().GetRiskFreeRate()==static_cast<float>(0.1)
			&& ppt3.GetPathComponent(i).GetInputOptionData().GetStrikePrice()==static_cast<float>(110.)
			&& ppt3.GetPathComponent(i).GetInputOptionData().GetNumberOfIntervals()==static_cast<unsigned int>(365)
			&& ppt3.GetPathComponent(i).GetInputOptionData().GetTimeToMaturity()==static_cast<float>(365.)
			&& ppt3.GetPathComponent(i).GetInputOptionData().GetDeltaTime()==static_cast<float>(1.)
			&& ppt3.GetPathComponent(i).GetInputOptionData().GetOptionType()==static_cast<char>('c')
		);
		
		cout << tests[i] << "\t";
	}
	cout << endl;
	cout << ppt3.GetIsCorrectDimension() << endl << endl;
	
	// Quando hai voglia implementa i check su market e option
	cout << "Methods testing" << endl;
	Path* other_paths = new Path[10];
	for(int i=0; i<10; ++i){
		other_paths[i].SetGaussianRandomVariable(0.5);
		other_paths[i].SetSpotPrice(24.);
	}
	
	ppt2.SetPathArray(10, other_paths);
	test_dim = (ppt2.GetNumberOfPathsPerThread() == static_cast<unsigned int>(10));
	cout << test_dim << endl;
	for(unsigned int i=0; i<ppt2.GetNumberOfPathsPerThread(); ++i){		
		tests[i] = (ppt2.GetPathComponent(i).GetGaussianRandomVariable() == static_cast<float>(0.5)
			&& ppt2.GetPathComponent(i).GetSpotPrice() == static_cast<float>(24.));
		
		cout << tests[i] << "\t";
	}
	cout << endl;
	cout << ppt2.GetIsCorrectDimension() << endl << endl;
	
		
	Path third_path(MarketData, OptionData, 100.);
	ppt2.SetPathComponent(3, third_path);
	test_dim = (ppt2.GetNumberOfPathsPerThread() == static_cast<unsigned int>(10));
	cout << test_dim << endl;
	tests[3] = (ppt2.GetPathComponent(3).GetGaussianRandomVariable() == static_cast<float>(0.)
			&& ppt2.GetPathComponent(3).GetSpotPrice() == static_cast<float>(100.));
	cout << tests[3] << endl;
	
	ppt2.SetPathArray(9, other_paths);
	cout << !(ppt2.GetIsCorrectDimension()) << endl << endl;
	

	return 0;

}
