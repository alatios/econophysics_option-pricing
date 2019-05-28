# econophysics_option-pricing
Econophysics CUDA project simulating a gaussian option pricing model with a Monte Carlo approach. Computes European style put/call options with fixed time to maturity, comparing the output to the theoretical result from the Black-Scholes formula.

## Tree
```
econophysics_option-pricing/
├── [4.7K]  README.md
├── [ 512]  libraries
│   ├── [ 512]  InputGPUData
│   │   ├── [1.4K]  Input_gpu_data.cu
│   │   ├── [ 821]  Input_gpu_data.cuh
│   │   ├── [8.2K]  Input_gpu_data.o
│   │   ├── [1.5K]  UnitTest.cu
│   │   ├── [7.8K]  UnitTest.o
│   │   ├── [563K]  UnitTest.x
│   │   ├── [  15]  clean.sh
│   │   └── [ 347]  makefile
│   ├── [ 512]  InputMCData
│   │   ├── [2.0K]  Input_MC_data.cu
│   │   ├── [ 975]  Input_MC_data.cuh
│   │   ├── [ 13K]  Input_MC_data.o
│   │   ├── [2.9K]  UnitTest.cu
│   │   ├── [ 11K]  UnitTest.o
│   │   ├── [577K]  UnitTest.x
│   │   ├── [  15]  clean.sh
│   │   └── [ 379]  makefile
│   ├── [ 512]  InputMarketData
│   │   ├── [1.3K]  Input_market_data.cu
│   │   ├── [ 915]  Input_market_data.cuh
│   │   ├── [8.8K]  Input_market_data.o
│   │   ├── [1.7K]  UnitTest.cu
│   │   ├── [8.9K]  UnitTest.o
│   │   ├── [563K]  UnitTest.x
│   │   ├── [  15]  clean.sh
│   │   └── [ 362]  makefile
│   ├── [ 512]  InputOptionData
│   │   ├── [2.1K]  Input_option_data.cu
│   │   ├── [1.2K]  Input_option_data.cuh
│   │   ├── [ 12K]  Input_option_data.o
│   │   ├── [2.3K]  UnitTest.cu
│   │   ├── [9.7K]  UnitTest.o
│   │   ├── [567K]  UnitTest.x
│   │   ├── [  15]  clean.sh
│   │   └── [ 350]  makefile
│   ├── [ 512]  OutputMCData
│   │   ├── [4.4K]  Output_MC_data.cu
│   │   ├── [1.6K]  Output_MC_data.cuh
│   │   ├── [ 99K]  Output_MC_data.o
│   │   ├── [3.4K]  UnitTest.cu
│   │   ├── [ 15K]  UnitTest.o
│   │   ├── [699K]  UnitTest.x
│   │   ├── [  15]  clean.sh
│   │   └── [ 517]  makefile
│   ├── [ 512]  Path
│   │   ├── [2.5K]  Path.cu
│   │   ├── [1.2K]  Path.cuh
│   │   ├── [ 18K]  Path.o
│   │   ├── [4.5K]  UnitTest.cu
│   │   ├── [ 18K]  UnitTest.o
│   │   ├── [595K]  UnitTest.x
│   │   ├── [  15]  clean.sh
│   │   └── [ 499]  makefile
│   ├── [ 512]  PathPerThread
│   │   ├── [2.4K]  Path_per_thread.cu
│   │   ├── [ 927]  Path_per_thread.cuh
│   │   ├── [ 17K]  Path_per_thread.o
│   │   ├── [5.5K]  UnitTest.cu
│   │   ├── [ 19K]  UnitTest.o
│   │   ├── [613K]  UnitTest.x
│   │   ├── [  15]  clean.sh
│   │   └── [ 490]  makefile
│   └── [ 283]  unit_test.sh
├── [ 512]  price_computing
│   ├── [ 830]  main.cu
│   ├── [5.2K]  main.o
│   ├── [765K]  main.x
│   └── [1.1K]  makefile
└── [ 512]  random_generator
    ├── [ 245]  Makefile
    ├── [4.1K]  main.cu
    ├── [ 23K]  main.o
    ├── [591K]  main.x
    ├── [4.7M]  out_aptly.dat
    ├── [3.5K]  rng.cu
    ├── [1.6K]  rng.cuh
    └── [ 19K]  rng.o

10 directories, 70 files
```
