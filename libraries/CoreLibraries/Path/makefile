FLAGS=-std=c++11 -Wno-deprecated-gpu-targets
TARGS_PATH=../../InputStructures/InputOptionData/Input_option_data.o Path.o UnitTest.o
TARGS_BS= \
../../InputStructures/InputGPUData/Input_gpu_data.o \
../../InputStructures/InputMCData/Input_MC_data.o \
../../InputStructures/InputOptionData/Input_option_data.o \
../../CoreLibraries/DataStreamManager/Data_stream_manager.o \
../../CoreLibraries/Statistics/Statistics.o \
Path.o \
../../CoreLibraries/RandomGenerator/RNG.o \
../../CoreLibraries/SupportFunctions/Support_functions.o \
../../OutputStructures/OutputMCData/Output_MC_data.o \
BlackScholesUnitTest.o

NVCC=nvcc

ECHO=/bin/echo

all: path bs

path: $(TARGS_PATH)
	$(NVCC) $(FLAGS) $(TARGS_PATH) -o UnitTest.x

bs: $(TARGS_BS)
	$(NVCC) $(FLAGS) $(TARGS_BS) -o BlackScholesUnitTest.x

%.o: %.cu
	$(NVCC) $(FLAGS) -dc $< -o $@

clean:
	@(cd ../../InputStructures/InputGPUData && rm -f *.x *.o) 		|| ($(ECHO) "Failed to clean ../../InputStructures/InputGPUData." && exit 1)
	@(cd ../../InputStructures/InputMarketData && rm -f *.x *.o)	|| ($(ECHO) "Failed to clean ../../InputStructures/InputMarketData." && exit 1)
	@(cd ../../InputStructures/InputMCData && rm -f *.x *.o)		|| ($(ECHO) "Failed to clean ../../InputStructures/InputMCData." && exit 1)
	@(cd ../../InputStructures/InputOptionData && rm -f *.x *.o)	|| ($(ECHO) "Failed to clean ../../InputStructures/InputOptionData." && exit 1)
	@(cd ../../CoreLibraries/DataStreamManager && rm -f *.x *.o)	|| ($(ECHO) "Failed to clean ../../CoreLibraries/DataStreamManager." && exit 1)
	@(cd ../../CoreLibraries/Statistics && rm -f *.x *.o)			|| ($(ECHO) "Failed to clean ../../CoreLibraries/Statistics." && exit 1)
	@(cd ../../CoreLibraries/RandomGenerator && rm -f *.x *.o)		|| ($(ECHO) "Failed to clean ../../CoreLibraries/RandomGenerator." && exit 1)
	@(cd ../../CoreLibraries/SupportFunctions && rm -f *.x *.o)		|| ($(ECHO) "Failed to clean ../../CoreLibraries/SupportFunctions." && exit 1)
	@(cd ../../OutputStructures/OutputMCData && rm -f *.x *.o )		|| ($(ECHO) "Failed to clean ../../OutputStructures/OutputMCData." && exit 1)
	@rm -f *.x *.o 														|| ($(ECHO) "Failed to clean root directory." && exit 1)

	@rm -f *.x *.o && echo "Done cleaning."

run: clean
	@make --no-print-directory
	@$(ECHO) "Running Path unit test..."
	@./UnitTest.x
	@$(ECHO) "Running Black and Scholes unit test..."
	@./BlackScholesUnitTest.x
