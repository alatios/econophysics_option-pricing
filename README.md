# econophysics_option-pricing
Econophysics CUDA project simulating a gaussian option pricing model with a Monte Carlo approach. Computes European style put/call options with fixed time to maturity, comparing the output to the theoretical result from the Black-Scholes formula.

## Tree
```
econophysics_option-pricing/
├── [2.5K]  README.md
├── [ 512]  libraries
│   ├── [ 512]  InputGPUData
│   │   ├── [1.0K]  Input_gpu_data.cu
│   │   ├── [ 719]  Input_gpu_data.cuh
│   │   ├── [ 974]  UnitTest.cu
│   │   ├── [  15]  clean.sh
│   │   └── [ 369]  makefile
│   ├── [ 512]  InputMCData
│   │   ├── [ 837]  Input_MC_data.cu
│   │   ├── [ 636]  Input_MC_data.cuh
│   │   ├── [ 890]  UnitTest.cu
│   │   ├── [  15]  clean.sh
│   │   └── [ 408]  makefile
│   ├── [ 512]  InputMarketData
│   │   ├── [1.3K]  Input_market_data.cu
│   │   ├── [ 938]  Input_market_data.cuh
│   │   ├── [1.7K]  UnitTest.cu
│   │   ├── [  15]  clean.sh
│   │   └── [ 375]  makefile
│   ├── [ 512]  InputOptionData
│   │   ├── [2.0K]  Input_option_data.cu
│   │   ├── [1.1K]  Input_option_data.cuh
│   │   ├── [2.0K]  UnitTest.cu
│   │   ├── [  15]  clean.sh
│   │   └── [ 375]  makefile
│   ├── [ 512]  OutputMCData
│   │   ├── [3.8K]  Output_MC_data.cu
│   │   ├── [1.7K]  Output_MC_data.cuh
│   │   ├── [2.8K]  UnitTest.cu
│   │   ├── [  15]  clean.sh
│   │   └── [ 444]  makefile
│   ├── [ 512]  OutputMCPerThread
│   │   ├── [2.6K]  Output_MC_per_thread.cu
│   │   ├── [1.7K]  Output_MC_per_thread.cuh
│   │   ├── [2.1K]  UnitTest.cu
│   │   ├── [  15]  clean.sh
│   │   └── [ 381]  makefile
│   ├── [ 512]  Path
│   │   ├── [2.4K]  Path.cu
│   │   ├── [1.2K]  Path.cuh
│   │   ├── [2.3K]  UnitTest.cu
│   │   ├── [  15]  clean.sh
│   │   └── [ 413]  makefile
│   └── [ 266]  unit_test.sh
├── [7.8K]  main.cu
├── [1.7K]  makefile
├── [ 512]  new_rng
│   ├── [ 273]  Makefile
│   ├── [9.3K]  main.cu
│   ├── [4.0K]  rng.cu
│   └── [1.9K]  rng.cuh
├── [ 512]  python_analysis_scripts
│   ├── [1004]  AnalyzeFinalSpotPriceDistribution.py
│   ├── [ 987]  AnalyzePayoffDistribution.py
│   └── [2.4K]  AnalyzeSpotPriceEvolution.py
└── [ 512]  random_generator
    ├── [ 12K]  main.cu
    ├── [ 233]  makefile
    ├── [3.5K]  rng.cu
    └── [1.9K]  rng.cuh

11 directories, 50 files
```
