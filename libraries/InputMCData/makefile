FLAGS=-std=c++11 -Wno-deprecated-gpu-targets
TARGS=../InputGPUData/Input_gpu_data.o Input_MC_data.o UnitTest.o
NVCC=nvcc

ECHO=/bin/echo

all: build

build: $(TARGS)
	$(NVCC) $(FLAGS) $(TARGS) -o UnitTest.x

%.o: %.cu
	$(NVCC) $(FLAGS) $(LIBS) -dc $< -o $@

clean:
	@rm -f *.x *.o && echo "Done cleaning."

run: clean
	@make --no-print-directory
	@$(ECHO) "Running Input_MC_data unit test..."
	@./UnitTest.x
