# Econophysics option pricing project "Orbison"
Econophysics option pricing CUDA project simulating evolutions of the underlying asset through a gaussian Monte Carlo approach. Computes European-style forward contracts, call/put plain vanilla and performance corridor options with given time to maturity and key parameters. Includes sample Bash and Python 3 scripts for parameter analysis automation.

## Tree structure
```
econophysics_option-pricing/
├── LICENSE
├── README.md
├── analysis_scripts
│   ├── ComputationTimeStudies_Tesla_AnalyzeResults.py
│   ├── ComputationTimeStudies_Tesla_GatherResults.sh
│   ├── ComputationTimeStudies_Tesla_RunSimulations.sh
│   ├── CorrelationTests_AnalyzeResults.py
│   ├── CorrelationTests_ProcessAndGatherResults.sh
│   ├── GainFactor_AnalyzeResults.py
│   ├── GainFactor_GatherResults.sh
│   ├── GainFactor_RunSimulations.sh
│   ├── InfiniteIntervals_GatherResults.sh
│   ├── InfiniteIntervals_RunSimulations.sh
│   ├── NegativePrices_AnalyzeResults.py
│   ├── NegativePrices_GatherResults.sh
│   ├── NegativePrices_RunSimulations.sh
│   ├── OptionPriceBimodal_AnalyzeResults.py
│   ├── OptionPriceBimodal_GatherResults.sh
│   ├── OptionPriceBimodal_RunSimulations.sh
│   ├── OptionPriceVsB_AnalyzeResults.py
│   ├── OptionPriceVsB_GatherResults.sh
│   ├── OptionPriceVsB_RunSimulations.sh
│   ├── OptionPriceVsM_AnalyzeResults.py
│   ├── OptionPriceVsM_GatherResults.sh
│   └── OptionPriceVsM_RunSimulations.sh
├── input.dat
├── input.dat.template
├── inputs
│   ├── ComputationTimeStudies_Tesla
│   ├── GainFactor_Tesla
│   ├── InfiniteIntervals
│   ├── NegativePrices
│   ├── OptionPriceBimodal
│   ├── OptionPriceVsB
│   └── OptionPriceVsM
├── libraries
│   ├── CoreLibraries
│   │   ├── DataStreamManager
│   │   │   ├── Data_stream_manager.cu
│   │   │   ├── Data_stream_manager.cuh
│   │   │   ├── UnitTest.cu
│   │   │   ├── input.dat
│   │   │   └── makefile
│   │   ├── Path
│   │   │   ├── BlackScholesUnitTest.cu
│   │   │   ├── Path.cu
│   │   │   ├── Path.cuh
│   │   │   ├── UnitTest.cu
│   │   │   ├── clean.sh
│   │   │   ├── input.dat
│   │   │   └── makefile
│   │   ├── RandomGenerator
│   │   │   ├── CorrelationTest.cu
│   │   │   ├── CorrelationTests_Autocorrelations.dat
│   │   │   ├── CorrelationTests_InterstreamAutocorrelations.dat
│   │   │   ├── CorrelationTests_MainIntraStream.dat
│   │   │   ├── CorrelationTests_UnprocessedOutput.dat
│   │   │   ├── CorrelationToolbox.cu
│   │   │   ├── CorrelationToolbox.cuh
│   │   │   ├── OutputTest.cu
│   │   │   ├── RNG.cu
│   │   │   ├── RNG.cuh
│   │   │   ├── graphs
│   │   │   │   ├── CorrelationTests_GaussInterStreamAutocorrelationHistogram.pdf
│   │   │   │   ├── CorrelationTests_GaussInterStreamAutocorrelationVsOffset.pdf
│   │   │   │   ├── CorrelationTests_InterStreamCorrelations.pdf
│   │   │   │   ├── CorrelationTests_IntraStreamCorrelations.pdf
│   │   │   │   ├── CorrelationTests_KurtosisVsVariance.pdf
│   │   │   │   ├── CorrelationTests_UniAvgVsGaussAvg.pdf
│   │   │   │   ├── CorrelationTests_UniformInterStreamAutocorrelationHistogram.pdf
│   │   │   │   └── CorrelationTests_UniformInterStreamAutocorrelationVsOffset.pdf
│   │   │   └── makefile
│   │   ├── Statistics
│   │   │   ├── Statistics.cu
│   │   │   ├── Statistics.cuh
│   │   │   ├── UnitTest.cu
│   │   │   ├── clean.sh
│   │   │   └── makefile
│   │   └── SupportFunctions
│   │       ├── Support_functions.cu
│   │       └── Support_functions.cuh
│   ├── InputStructures
│   │   ├── InputGPUData
│   │   │   ├── Input_gpu_data.cu
│   │   │   ├── Input_gpu_data.cuh
│   │   │   ├── UnitTest.cu
│   │   │   ├── clean.sh
│   │   │   └── makefile
│   │   ├── InputMCData
│   │   │   ├── Input_MC_data.cu
│   │   │   ├── Input_MC_data.cuh
│   │   │   ├── UnitTest.cu
│   │   │   ├── clean.sh
│   │   │   └── makefile
│   │   ├── InputMarketData
│   │   │   ├── Input_market_data.cuh
│   │   │   ├── UnitTest.cu
│   │   │   ├── clean.sh
│   │   │   └── makefile
│   │   └── InputOptionData
│   │       ├── Input_option_data.cu
│   │       ├── Input_option_data.cuh
│   │       ├── UnitTest.cu
│   │       ├── clean.sh
│   │       └── makefile
│   ├── OutputStructures
│   │   └── OutputMCData
│   │       ├── Output_MC_data.cu
│   │       ├── Output_MC_data.cuh
│   │       ├── UnitTest.cu
│   │       ├── clean.sh
│   │       └── makefile
│   └── unit_test.sh
├── main.cu
├── makefile
├── outputs
│   ├── ComputationTimeStudies_Tesla
│   ├── CorrelationTests
│   ├── GainFactor_Tesla
│   ├── InfiniteIntervals
│   ├── NegativePrices
│   ├── OptionPriceBimodal
│   ├── OptionPriceVsB
│   └── OptionPriceVsM
└── schematics
    ├── RecapMainCuDiagram.drawio
    └── RecapMainCuDiagram.pdf

41 directories, 2447 files
```
