#include "Input_market_data.cuh"
#include <iostream>

using namespace std;

int main(){

    Input_market_data market_data_1;

    market_data_1.InitialPrice = 1.;
    market_data_1.Volatility = 2.;
    market_data_1.RiskFreeRate = 3.;

    cout << "\n-------------Input_market_test-------------\n";
    
    bool test;
    test = (market_data_1.InitialPrice == static_cast<double>(1.));
    cout << test << "\t";
    test = (market_data_1.Volatility == static_cast<double>(2.));
    cout << test << "\t";
    test = (market_data_1.RiskFreeRate == static_cast<double>(3.));
    cout << test << "\n";

    return 0;
}
