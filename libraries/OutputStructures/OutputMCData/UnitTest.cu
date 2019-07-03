#include <iostream>

#include "Output_MC_data.cuh"

using namespace std;

int main(){

    Output_MC_data MC_data_1;

    MC_data_1.EstimatedPriceMCEuler = 1.;
    MC_data_1.ErrorMCEuler = 2.;
    MC_data_1.EstimatedPriceMCExact = 3.;
    MC_data_1.ErrorMCExact = 4.;
    MC_data_1.Tick = 5.;

    cout << "\n-------------Output_MC_test-------------\n";
    
    bool test;
    test = (MC_data_1.EstimatedPriceMCEuler == static_cast<double>(1.));
    cout << test << "\t";
    test = (MC_data_1.ErrorMCEuler == static_cast<double>(2.));
    cout << test << "\t";
    test = (MC_data_1.EstimatedPriceMCExact == static_cast<double>(3.));
    cout << test << "\t";
    test = (MC_data_1.ErrorMCExact == static_cast<double>(4.));
    cout << test << "\t";
    test = (MC_data_1.Tick == static_cast<double>(5.));
    cout << test << "\n";

    return 0;
}

