LIBS=
FLAGS=-std=c++11 -Wno-deprecated-gpu-targets
TARGS=libraries/InputGPUData/Input_gpu_data.o libraries/InputMarketData/Input_market_data.o libraries/InputMCData/Input_MC_data.o libraries/InputOptionData/Input_option_data.o libraries/OutputMCData/Output_MC_data.o libraries/Path/Path.o libraries/PathPerThread/Path_per_thread.o random_generator/rng.o main.o
NVCC=nvcc

ECHO=/bin/echo

all: $(TARGS)
	$(NVCC) $(FLAGS) $(LIBS) $(TARGS) -o main.x

obj: %.o

%.o: %.cu
	$(NVCC) $(FLAGS) $(LIBS) -dc $< -o $@

clean:
	@cd libraries/InputGPUData && (rm -f *.x *.o || echo "Failed to clean libraries/InputGPUData.")
	@cd libraries/InputMarketData && rm -f *.x *.o
	@cd libraries/InputMCData && rm -f *.x *.o
	@cd libraries/InputOptionData && rm -f *.x *.o
	@cd libraries/OutputMCData && rm -f *.x *.o
	@cd libraries/Path && rm -f *.x *.o
	@cd libraries/PathPerThread && rm -f *.x *.o
	@cd random_generator && rm -f *.x *.o
	@rm -f *.x *.o
	@echo "Done cleaning."

run:
	make
	./main.x
