FLAGS=-std=c++11 -Wno-deprecated-gpu-targets

TARGS= \
libraries/InputStructures/InputGPUData/Input_gpu_data.o \
libraries/InputStructures/InputMCData/Input_MC_data.o \
libraries/InputStructures/InputOptionData/Input_option_data.o \
libraries/CoreLibraries/Statistics/Statistics.o \
libraries/CoreLibraries/Path/Path.o \
libraries/CoreLibraries/RandomGenerator/rng.o \
libraries/CoreLibraries/SupportFunctions/Support_functions.o \
main.o

NVCC=nvcc

ECHO=/bin/echo

all: $(TARGS)
	$(NVCC) $(FLAGS) $(TARGS) -o main.x

%.o: %.cu
	$(NVCC) $(FLAGS) -dc $< -o $@

clean:
	@(cd libraries/InputStructures/InputGPUData && rm -f *.x *.o) 		|| ($(ECHO) "Failed to clean libraries/InputStructures/InputGPUData." && exit 1)
	@(cd libraries/InputStructures/InputMarketData && rm -f *.x *.o)	|| ($(ECHO) "Failed to clean libraries/InputStructures/InputMarketData." && exit 1)
	@(cd libraries/InputStructures/InputMCData && rm -f *.x *.o)		|| ($(ECHO) "Failed to clean libraries/InputStructures/InputMCData." && exit 1)
	@(cd libraries/InputStructures/InputOptionData && rm -f *.x *.o)	|| ($(ECHO) "Failed to clean libraries/InputStructures/InputOptionData." && exit 1)
	@(cd libraries/CoreLibraries/Statistics && rm -f *.x *.o)			|| ($(ECHO) "Failed to clean libraries/CoreLibraries/Statistics." && exit 1)
	@(cd libraries/CoreLibraries/Path && rm -f *.x *.o)					|| ($(ECHO) "Failed to clean libraries/CoreLibraries/Path." && exit 1)
	@(cd libraries/CoreLibraries/RandomGenerator && rm -f *.x *.o)		|| ($(ECHO) "Failed to clean libraries/CoreLibraries/RandomGenerator." && exit 1)
	@(cd libraries/CoreLibraries/SupportFunctions && rm -f *.x *.o)		|| ($(ECHO) "Failed to clean libraries/CoreLibraries/SupportFunctions." && exit 1)
	@(cd libraries/OutputStructures/OutputMCData && rm -f *.x *.o )		|| ($(ECHO) "Failed to clean libraries/OutputStructures/OutputMCData." && exit 1)
	@rm -f *.x *.o 														|| ($(ECHO) "Failed to clean root directory." && exit 1)
	@$(ECHO) "Done cleaning."

run:
	make --no-print-directory
	./main.x
