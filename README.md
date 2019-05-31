# econophysics_option-pricing
Econophysics CUDA project simulating a gaussian option pricing model with a Monte Carlo approach. Computes European style put/call options with fixed time to maturity, comparing the output to the theoretical result from the Black-Scholes formula.

## Tree
```
econophysics_option-pricing/
├── README.md
├── libraries
│   ├── InputGPUData
│   │   ├── Input_gpu_data.cu
│   │   ├── Input_gpu_data.cuh
│   │   ├── UnitTest.cu
│   │   ├── clean.sh
│   │   └── makefile
│   ├── InputMCData
│   │   ├── Input_MC_data.cu
│   │   ├── Input_MC_data.cuh
│   │   ├── UnitTest.cu
│   │   ├── clean.sh
│   │   └── makefile
│   ├── InputMarketData
│   │   ├── Input_market_data.cu
│   │   ├── Input_market_data.cuh
│   │   ├── UnitTest.cu
│   │   ├── clean.sh
│   │   └── makefile
│   ├── InputOptionData
│   │   ├── Input_option_data.cu
│   │   ├── Input_option_data.cuh
│   │   ├── UnitTest.cu
│   │   ├── clean.sh
│   │   └── makefile
│   ├── OutputMCData
│   │   ├── Output_MC_data.cu
│   │   ├── Output_MC_data.cuh
│   │   ├── UnitTest.cu
│   │   ├── clean.sh
│   │   └── makefile
│   ├── Path
│   │   ├── Path.cu
│   │   ├── Path.cuh
│   │   ├── UnitTest.cu
│   │   ├── clean.sh
│   │   └── makefile
│   ├── PathPerThread
│   │   ├── Path_per_thread.cu
│   │   ├── Path_per_thread.cuh
│   │   ├── UnitTest.cu
│   │   ├── clean.sh
│   │   └── makefile
│   └── unit_test.sh
├── main.cu
├── makefile
└── random_generator
    ├── Makefile
    ├── main.cu
    ├── out_aptly.dat
    ├── rng.cu
    └── rng.cuh

9 directories, 44 files
```
