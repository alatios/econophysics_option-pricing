# Econophysics option pricing project "Orbison"
Econophysics option pricing CUDA project simulating evolutions of the underlying asset through a gaussian Monte Carlo approach. Computes European-style plain vanilla put/call options with fixed time to maturity and strike price, comparing the estimated payoff to the theoretical result obtained from the Black-Scholes formula.

## Tree structure
```
econophysics_option-pricing/
├── [ 671]  input.dat
├── [4.0K]  libraries
│   ├── [4.0K]  CoreLibraries
│   │   ├── [4.0K]  Path
│   │   │   ├── [  15]  clean.sh
│   │   │   ├── [ 413]  makefile
│   │   │   ├── [4.1K]  Path.cu
│   │   │   ├── [1.8K]  Path.cuh
│   │   │   └── [1.9K]  UnitTest.cu
│   │   ├── [4.0K]  RandomGenerator
│   │   │   ├── [9.6K]  main.cu
│   │   │   ├── [ 233]  makefile
│   │   │   ├── [2.5K]  rng.cu
│   │   │   └── [1.7K]  rng.cuh
│   │   ├── [4.0K]  Statistics
│   │   │   ├── [  15]  clean.sh
│   │   │   ├── [ 361]  makefile
│   │   │   ├── [1.7K]  Statistics.cu
│   │   │   ├── [1.1K]  Statistics.cuh
│   │   │   └── [2.0K]  UnitTest.cu
│   │   └── [4.0K]  SupportFunctions
│   │       ├── [5.6K]  Support_functions.cu
│   │       └── [1.5K]  Support_functions.cuh
│   ├── [4.0K]  InputStructures
│   │   ├── [4.0K]  InputGPUData
│   │   │   ├── [  15]  clean.sh
│   │   │   ├── [ 323]  Input_gpu_data.cu
│   │   │   ├── [ 341]  Input_gpu_data.cuh
│   │   │   ├── [ 369]  makefile
│   │   │   └── [1.1K]  UnitTest.cu
│   │   ├── [4.0K]  InputMarketData
│   │   │   ├── [  15]  clean.sh
│   │   │   ├── [ 268]  Input_market_data.cuh
│   │   │   ├── [ 375]  makefile
│   │   │   └── [1.6K]  UnitTest.cu
│   │   ├── [4.0K]  InputMCData
│   │   │   ├── [  15]  clean.sh
│   │   │   ├── [ 299]  Input_MC_data.cu
│   │   │   ├── [ 293]  Input_MC_data.cuh
│   │   │   └── [ 408]  makefile
│   │   └── [4.0K]  InputOptionData
│   │       ├── [  15]  clean.sh
│   │       ├── [ 254]  Input_option_data.cu
│   │       ├── [ 708]  Input_option_data.cuh
│   │       ├── [ 375]  makefile
│   │       └── [2.3K]  UnitTest.cu
│   ├── [4.0K]  OutputStructures
│   │   └── [4.0K]  OutputMCData
│   │       ├── [  15]  clean.sh
│   │       ├── [ 444]  makefile
│   │       ├── [ 273]  Output_MC_data.cuh
│   │       └── [1.9K]  UnitTest.cu
│   └── [ 259]  unit_test.sh
├── [ 34K]  LICENSE
├── [5.1K]  main.cu
├── [1.7K]  makefile
├── [4.0K]  new_rng
│   ├── [9.3K]  main.cu
│   ├── [ 273]  Makefile
│   ├── [4.0K]  rng.cu
│   └── [1.9K]  rng.cuh
├── [4.0K]  python_analysis_scripts
│   ├── [1004]  AnalyzeFinalSpotPriceDistribution.py
│   ├── [ 987]  AnalyzePayoffDistribution.py
│   └── [2.4K]  AnalyzeSpotPriceEvolution.py
└── [3.0K]  README.md

15 directories, 51 files
```
