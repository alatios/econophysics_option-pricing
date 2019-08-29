# Econophysics option pricing project "Orbison"
Econophysics option pricing CUDA project simulating evolutions of the underlying asset through a gaussian Monte Carlo approach. Computes European-style forward contracts, call/put plain vanilla and performance corridor options with given time to maturity and key parameters. Includes sample Bash and Python 3 scripts for parameter analysis automation.

## Tree structure
```
econophysics_option-pricing/
├── [ 34K]  LICENSE
├── [3.5K]  README.md
├── [ 512]  analysis_scripts
│   ├── [1.8K]  ComputationTimeStudies_Tesla_AnalyzeResults.py
│   ├── [ 428]  ComputationTimeStudies_Tesla_GatherResults.sh
│   ├── [1.1K]  ComputationTimeStudies_Tesla_RunSimulations.sh
│   ├── [4.7K]  CorrelationTests_AnalyzeResults.py
│   ├── [2.9K]  CorrelationTests_ProcessAndGatherResults.sh
│   ├── [1.2K]  GainFactor_AnalyzeResults.py
│   ├── [ 798]  GainFactor_GatherResults.sh
│   ├── [1.1K]  GainFactor_RunSimulations.sh
│   ├── [1.0K]  InfiniteIntervals_GatherResults.sh
│   ├── [1.1K]  InfiniteIntervals_RunSimulations.sh
│   ├── [1.6K]  NegativePrices_AnalyzeResults.py
│   ├── [ 364]  NegativePrices_GatherResults.sh
│   ├── [1.1K]  NegativePrices_RunSimulations.sh
│   ├── [1.0K]  OptionPriceBimodal_AnalyzeResults.py
│   ├── [ 605]  OptionPriceBimodal_GatherResults.sh
│   ├── [1.1K]  OptionPriceBimodal_RunSimulations.sh
│   ├── [2.5K]  OptionPriceVsB_AnalyzeResults.py
│   ├── [ 790]  OptionPriceVsB_GatherResults.sh
│   ├── [1.2K]  OptionPriceVsB_RunSimulations.sh
│   ├── [6.1K]  OptionPriceVsM_AnalyzeResults.py
│   ├── [ 772]  OptionPriceVsM_GatherResults.sh
│   └── [1.2K]  OptionPriceVsM_RunSimulations.sh
├── [1.2K]  input.dat
├── [1.3K]  input.dat.template
├── [ 512]  inputs
│   ├── [ 512]  ComputationTimeStudies_Tesla
│   ├── [ 512]  GainFactor_Tesla
│   ├── [ 512]  InfiniteIntervals
│   ├── [ 512]  NegativePrices
│   ├── [ 512]  OptionPriceBimodal
│   ├── [ 512]  OptionPriceVsB
│   └── [ 512]  OptionPriceVsM
├── [ 512]  libraries
│   ├── [ 512]  CoreLibraries
│   │   ├── [ 512]  DataStreamManager
│   │   │   ├── [5.9K]  Data_stream_manager.cu
│   │   │   ├── [1.3K]  Data_stream_manager.cuh
│   │   │   ├── [2.1K]  UnitTest.cu
│   │   │   ├── [1.2K]  input.dat
│   │   │   └── [ 625]  makefile
│   │   ├── [ 512]  Path
│   │   │   ├── [2.6K]  BlackScholesUnitTest.cu
│   │   │   ├── [5.1K]  Path.cu
│   │   │   ├── [1.8K]  Path.cuh
│   │   │   ├── [5.4K]  UnitTest.cu
│   │   │   ├── [  15]  clean.sh
│   │   │   ├── [1.2K]  input.dat
│   │   │   └── [2.3K]  makefile
│   │   ├── [ 512]  RandomGenerator
│   │   │   ├── [7.9K]  CorrelationTest.cu
│   │   │   ├── [663K]  CorrelationTests_Autocorrelations.dat
│   │   │   ├── [685K]  CorrelationTests_InterstreamAutocorrelations.dat
│   │   │   ├── [ 20K]  CorrelationTests_MainIntraStream.dat
│   │   │   ├── [2.4M]  CorrelationTests_UnprocessedOutput.dat
│   │   │   ├── [5.6K]  CorrelationToolbox.cu
│   │   │   ├── [2.2K]  CorrelationToolbox.cuh
│   │   │   ├── [9.7K]  OutputTest.cu
│   │   │   ├── [4.5K]  RNG.cu
│   │   │   ├── [2.2K]  RNG.cuh
│   │   │   ├── [ 512]  graphs
│   │   │   │   ├── [ 44K]  CorrelationTests_GaussInterStreamAutocorrelationHistogram.pdf
│   │   │   │   ├── [200K]  CorrelationTests_GaussInterStreamAutocorrelationVsOffset.pdf
│   │   │   │   ├── [291K]  CorrelationTests_InterStreamCorrelations.pdf
│   │   │   │   ├── [458K]  CorrelationTests_IntraStreamCorrelations.pdf
│   │   │   │   ├── [ 26K]  CorrelationTests_KurtosisVsVariance.pdf
│   │   │   │   ├── [ 27K]  CorrelationTests_UniAvgVsGaussAvg.pdf
│   │   │   │   ├── [ 43K]  CorrelationTests_UniformInterStreamAutocorrelationHistogram.pdf
│   │   │   │   └── [ 98K]  CorrelationTests_UniformInterStreamAutocorrelationVsOffset.pdf
│   │   │   └── [ 628]  makefile
│   │   ├── [ 512]  Statistics
│   │   │   ├── [2.2K]  Statistics.cu
│   │   │   ├── [1.3K]  Statistics.cuh
│   │   │   ├── [1.8K]  UnitTest.cu
│   │   │   ├── [  15]  clean.sh
│   │   │   └── [ 361]  makefile
│   │   └── [ 512]  SupportFunctions
│   │       ├── [8.1K]  Support_functions.cu
│   │       └── [1.4K]  Support_functions.cuh
│   ├── [ 512]  InputStructures
│   │   ├── [ 512]  InputGPUData
│   │   │   ├── [ 322]  Input_gpu_data.cu
│   │   │   ├── [ 302]  Input_gpu_data.cuh
│   │   │   ├── [ 482]  UnitTest.cu
│   │   │   ├── [  15]  clean.sh
│   │   │   └── [ 369]  makefile
│   │   ├── [ 512]  InputMCData
│   │   │   ├── [ 346]  Input_MC_data.cu
│   │   │   ├── [ 492]  Input_MC_data.cuh
│   │   │   ├── [ 990]  UnitTest.cu
│   │   │   ├── [  15]  clean.sh
│   │   │   └── [ 408]  makefile
│   │   ├── [ 512]  InputMarketData
│   │   │   ├── [ 244]  Input_market_data.cuh
│   │   │   ├── [ 618]  UnitTest.cu
│   │   │   ├── [  15]  clean.sh
│   │   │   └── [ 355]  makefile
│   │   └── [ 512]  InputOptionData
│   │       ├── [ 252]  Input_option_data.cu
│   │       ├── [ 550]  Input_option_data.cuh
│   │       ├── [1.6K]  UnitTest.cu
│   │       ├── [  15]  clean.sh
│   │       └── [ 375]  makefile
│   ├── [ 512]  OutputStructures
│   │   └── [ 512]  OutputMCData
│   │       ├── [ 546]  Output_MC_data.cu
│   │       ├── [ 483]  Output_MC_data.cuh
│   │       ├── [ 852]  UnitTest.cu
│   │       ├── [  15]  clean.sh
│   │       └── [ 369]  makefile
│   └── [ 350]  unit_test.sh
├── [1.6K]  main.cu
├── [2.2K]  makefile
├── [ 512]  outputs
│   ├── [ 512]  ComputationTimeStudies_Tesla
│   │   ├── [ 11K]  ComputationTime_Tesla_GatheredResults_1000SimsPerThread.dat
│   │   ├── [ 11K]  ComputationTime_Tesla_GatheredResults_2000SimsPerThread.dat
│   │   ├── [ 512]  graphs
│   │   │   ├── [ 19K]  ComputationTime_Tesla_GPUTimeVsNOfBlocks_VariousM_1000SimsPerThread.pdf
│   │   │   └── [ 19K]  ComputationTime_Tesla_GPUTimeVsNOfBlocks_VariousM_2000SimsPerThread.pdf
│   ├── [ 512]  CorrelationTests
│   │   ├── [663K]  CorrelationTests_Autocorrelations.dat
│   │   ├── [685K]  CorrelationTests_InterstreamAutocorrelations.dat
│   │   ├── [ 20K]  CorrelationTests_MainIntraStream.dat
│   │   └── [ 512]  graphs
│   │       ├── [ 45K]  CorrelationTests_GaussInterStreamAutocorrelationHistogram.pdf
│   │       ├── [200K]  CorrelationTests_GaussInterStreamAutocorrelationVsOffset.pdf
│   │       ├── [291K]  CorrelationTests_InterStreamCorrelations.pdf
│   │       ├── [458K]  CorrelationTests_IntraStreamCorrelations.pdf
│   │       ├── [ 26K]  CorrelationTests_KurtosisVsVariance.pdf
│   │       ├── [ 27K]  CorrelationTests_UniAvgVsGaussAvg.pdf
│   │       ├── [ 44K]  CorrelationTests_UniformInterStreamAutocorrelationHistogram.pdf
│   │       └── [ 98K]  CorrelationTests_UniformInterStreamAutocorrelationVsOffset.pdf
│   ├── [ 512]  GainFactor_Tesla
│   │   ├── [1.8K]  GainFactor_GatheredResults.dat
│   │   ├── [ 512]  graphs
│   │   │   └── [ 16K]  GainFactor_Tesla_GainFactorVsBlocks_VariousM.pdf
│   ├── [ 512]  InfiniteIntervals
│   │   ├── [6.7K]  InfiniteIntervals_GatheredResults.dat
│   ├── [ 512]  NegativePrices
│   │   ├── [ 655]  NegativePrices_GatheredResults.dat
│   │   ├── [ 512]  graphs
│   │   │   └── [ 16K]  NegativePrices_PercentageVsM_VariousSigmas.pdf
│   ├── [ 512]  OptionPriceBimodal
│   │   ├── [ 996]  OptionPriceBimodal_GatheredResults.dat
│   │   ├── [ 512]  graphs
│   │   │   └── [ 14K]  OptionPriceBimodal_PriceVsM.pdf
│   ├── [ 512]  OptionPriceVsB
│   │   ├── [ 512]  graphs
│   │   │   ├── [ 17K]  OptionPriceVsB_ExactErrorVsN_WithAllBs.pdf
│   │   │   ├── [ 13K]  OptionPriceVsB_HBVsB.pdf
│   │   │   └── [ 15K]  OptionPriceVsB_PriceVsB_N200mln.pdf
│   │   ├── [ 23K]  optionpricevsb_gatheredresults.dat
│   └── [ 512]  OptionPriceVsM
│       ├── [8.2K]  OptionPriceVsM_GatheredResults.dat
│       ├── [ 512]  graphs
│       │   ├── [ 18K]  OptionPriceVsM_DiscrepancyVsM_WithDifferentNs.pdf
│       │   ├── [ 17K]  OptionPriceVsM_EulerErrorVsM_WithDifferentNs.pdf
│       │   ├── [ 17K]  OptionPriceVsM_EulerPriceVsM_WithDifferentNs.pdf
│       │   ├── [ 19K]  OptionPriceVsM_ExactErrorVsM_N108.pdf
│       │   ├── [ 18K]  OptionPriceVsM_ExactErrorVsM_WithDifferentNs.pdf
│       │   ├── [ 17K]  OptionPriceVsM_ExactPriceVsM_WithDifferentNs.pdf
│       │   └── [ 14K]  OptionPriceVsM_PriceVsM_N100mln.pdf
├── [ 512]  report
│   ├── [ 12K]  01-introduzione.tex
│   ├── [ 40K]  02-risultati.tex
│   ├── [ 12K]  03-formuleesatte.tex
│   ├── [7.2K]  04-processistocastici.tex
│   ├── [3.3K]  05-cudaclassi.tex
│   ├── [ 89K]  99-appendix.tex
│   ├── [ 10K]  KTHEEtitlepage.sty
│   ├── [ 82K]  Landslide_Literature.bib
│   ├── [ 512]  graphs
│   │   ├── [ 19K]  ComputationTime_Tesla_GPUTimeVsNOfBlocks_VariousM_1000SimsPerThread.pdf
│   │   ├── [ 45K]  CorrelationTests_GaussInterStreamAutocorrelationHistogram.pdf
│   │   ├── [200K]  CorrelationTests_GaussInterStreamAutocorrelationVsOffset.pdf
│   │   ├── [291K]  CorrelationTests_InterStreamCorrelations.pdf
│   │   ├── [458K]  CorrelationTests_IntraStreamCorrelations.pdf
│   │   ├── [ 26K]  CorrelationTests_KurtosisVsVariance.pdf
│   │   ├── [ 27K]  CorrelationTests_UniAvgVsGaussAvg.pdf
│   │   ├── [ 44K]  CorrelationTests_UniformInterStreamAutocorrelationHistogram.pdf
│   │   ├── [ 98K]  CorrelationTests_UniformInterStreamAutocorrelationVsOffset.pdf
│   │   ├── [ 16K]  GainFactor_Tesla_GainFactorVsBlocks_VariousM.pdf
│   │   ├── [ 16K]  NegativePrices_PercentageVsM_VariousSigmas.pdf
│   │   ├── [ 14K]  OptionPriceBimodal_PriceVsM.pdf
│   │   ├── [ 17K]  OptionPriceVsB_ExactErrorVsN_WithAllBs.pdf
│   │   ├── [ 13K]  OptionPriceVsB_HBVsB.pdf
│   │   ├── [ 15K]  OptionPriceVsB_PriceVsB_N200mln.pdf
│   │   ├── [ 18K]  OptionPriceVsM_DiscrepancyVsM_WithDifferentNs.pdf
│   │   ├── [ 17K]  OptionPriceVsM_EulerErrorVsM_WithDifferentNs.pdf
│   │   ├── [ 17K]  OptionPriceVsM_EulerPriceVsM_WithDifferentNs.pdf
│   │   ├── [ 19K]  OptionPriceVsM_ExactErrorVsM_N108.pdf
│   │   ├── [ 18K]  OptionPriceVsM_ExactErrorVsM_WithDifferentNs.pdf
│   │   ├── [ 17K]  OptionPriceVsM_ExactPriceVsM_WithDifferentNs.pdf
│   │   ├── [ 14K]  OptionPriceVsM_PriceVsM_N100mln.pdf
│   │   └── [ 35K]  RecapMainCuDiagram.pdf
│   ├── [350K]  logo.png
│   ├── [3.3K]  main.tex
│   └── [ 512]  old
│       ├── [2.2K]  00-copertina.tex
│       ├── [2.4K]  main_backup.tex
│       └── [  75]  prefazione.tex
└── [ 512]  schematics
    ├── [1.9K]  RecapMainCuDiagram.drawio
    └── [ 35K]  RecapMainCuDiagram.pdf

44 directories, 2483 files
```
