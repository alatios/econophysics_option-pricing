FLAGS=-std=c++11 -Wno-deprecated-gpu-targets
TARGS= \
../../InputStructures/InputGPUData/Input_gpu_data.o \
../../InputStructures/InputMCData/Input_MC_data.o \
../../InputStructures/InputOptionData/Input_option_data.o \
../Statistics/Statistics.o \
../../OutputStructures/OutputMCData/Output_MC_data.o \
Data_stream_manager.o \
UnitTest.o
NVCC=nvcc

ECHO=/bin/echo

all: build

build: $(TARGS)
	$(NVCC) $(FLAGS) $(TARGS) -o UnitTest.x

%.o: %.cu
	$(NVCC) $(FLAGS) -dc $< -o $@

clean:
	@rm -f *.x *.o && echo "Done cleaning."

run: clean
	@make --no-print-directory
	@$(ECHO) "Running Statistics unit test..."
	@./UnitTest.x
