FLAGS=-std=c++11 -Wno-deprecated-gpu-targets

TARGS= \
libraries/InputGPUData/Input_gpu_data.o \
libraries/InputMarketData/Input_market_data.o \
libraries/InputMCData/Input_MC_data.o \
libraries/InputOptionData/Input_option_data.o \
libraries/OutputMCData/Output_MC_data.o \
libraries/Path/Path.o \
libraries/OutputMCPerThread/Output_MC_per_thread.o \
random_generator/rng.o \
main.o

NVCC=nvcc

ECHO=/bin/echo

all: $(TARGS)
	$(NVCC) $(FLAGS) $(TARGS) -o main.x

%.o: %.cu
	$(NVCC) $(FLAGS) -dc $< -o $@

clean:
	@(cd libraries/InputGPUData && rm -f *.x *.o) 		|| ($(ECHO) "Failed to clean libraries/InputGPUData." && exit 1)
	@(cd libraries/InputMarketData && rm -f *.x *.o)	|| ($(ECHO) "Failed to clean libraries/InputMarketData." && exit 1)
	@(cd libraries/InputMCData && rm -f *.x *.o)		|| ($(ECHO) "Failed to clean libraries/InputMCData." && exit 1)
	@(cd libraries/InputOptionData && rm -f *.x *.o)	|| ($(ECHO) "Failed to clean libraries/InputOptionData." && exit 1)
	@(cd libraries/OutputMCData && rm -f *.x *.o)		|| ($(ECHO) "Failed to clean libraries/OutputMCData." && exit 1)
	@(cd libraries/Path && rm -f *.x *.o)				|| ($(ECHO) "Failed to clean libraries/Path." && exit 1)
	@(cd libraries/OutputMCPerThread && rm -f *.x *.o )	|| ($(ECHO) "Failed to clean libraries/OutputMCPerThread." && exit 1)
	@(cd random_generator && rm -f *.x *.o)				|| ($(ECHO) "Failed to clean random_generator." && exit 1)
	@rm -f *.x *.o 										|| ($(ECHO) "Failed to clean root directory." && exit 1)
	@$(ECHO) "Done cleaning."

run:
	make --no-print-directory
	./main.x
