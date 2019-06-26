#ifndef __Output_MC_data_h__
#define __Output_MC_data_h__

#include <iostream>

struct Output_MC_data{
	
	double _EstimatedPriceMCEuler;
	double _ErrorMCEuler;
	double _EstimatedPriceMCExact;
	double _ErrorMCExact;
	double _Tick;							// Calculation time [ms]

};

#endif
