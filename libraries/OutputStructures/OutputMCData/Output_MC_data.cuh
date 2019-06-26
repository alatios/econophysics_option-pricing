#ifndef __Output_MC_data_h__
#define __Output_MC_data_h__

#include <iostream>

struct Output_MC_data{
	
	double EstimatedPriceMCEuler;
	double ErrorMCEuler;
	double EstimatedPriceMCExact;
	double ErrorMCExact;
	double Tick;							// Calculation time [ms]

};

#endif
