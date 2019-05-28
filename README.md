# econophysics_option-pricing
Econophysics CUDA project simulating a gaussian option pricing model with a Monte Carlo approach. Computes European style put/call options with fixed time to maturity, comparing the output to the theoretical result from the Black-Scholes formula.

## Tree
```
econophysics_option-pricing/
├── [3.7K]  README.md
├── [ 512]  libraries
│   ├── [ 512]  InputGPUData
│   │   ├── [1.4K]  Input_gpu_data.cu
│   │   ├── [ 821]  Input_gpu_data.cuh
│   │   ├── [1.5K]  UnitTest.cu
│   │   ├── [  15]  clean.sh
│   │   └── [ 347]  makefile
│   ├── [ 512]  InputMCData
│   │   ├── [2.0K]  Input_MC_data.cu
│   │   ├── [ 975]  Input_MC_data.cuh
│   │   ├── [2.9K]  UnitTest.cu
│   │   ├── [  15]  clean.sh
│   │   └── [ 379]  makefile
│   ├── [ 512]  InputMarketData
│   │   ├── [1.3K]  Input_market_data.cu
│   │   ├── [ 915]  Input_market_data.cuh
│   │   ├── [1.7K]  UnitTest.cu
│   │   ├── [  15]  clean.sh
│   │   └── [ 362]  makefile
│   ├── [ 512]  InputOptionData
│   │   ├── [2.1K]  Input_option_data.cu
│   │   ├── [1.2K]  Input_option_data.cuh
│   │   ├── [2.3K]  UnitTest.cu
│   │   ├── [  15]  clean.sh
│   │   └── [ 350]  makefile
│   ├── [ 512]  OutputMCData
│   │   ├── [4.4K]  Output_MC_data.cu
│   │   ├── [1.6K]  Output_MC_data.cuh
│   │   ├── [3.4K]  UnitTest.cu
│   │   ├── [  15]  clean.sh
│   │   └── [ 517]  makefile
│   ├── [ 512]  Path
│   │   ├── [2.5K]  Path.cu
│   │   ├── [1.2K]  Path.cuh
│   │   ├── [4.5K]  UnitTest.cu
│   │   ├── [  15]  clean.sh
│   │   └── [ 499]  makefile
│   ├── [ 512]  PathPerThread
│   │   ├── [2.4K]  Path_per_thread.cu
│   │   ├── [ 927]  Path_per_thread.cuh
│   │   ├── [5.5K]  UnitTest.cu
│   │   ├── [  15]  clean.sh
│   │   └── [ 490]  makefile
│   └── [ 283]  unit_test.sh
├── [ 512]  price_computing
│   ├── [ 830]  main.cu
│   └── [1.0K]  makefile
└── [ 512]  random_generator
    ├── [ 245]  Makefile
    ├── [4.1K]  main.cu
    ├── [4.7M]  out_aptly.dat
    ├── [3.5K]  rng.cu
    └── [1.6K]  rng.cuh

10 directories, 44 files
```
