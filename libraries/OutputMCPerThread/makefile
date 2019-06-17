FLAGS=-std=c++11 -Wno-deprecated-gpu-targets
TARGS=Output_MC_per_thread.o UnitTest.o
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
	@$(ECHO) "Running Output_MC_per_thread unit test..."
	@./UnitTest.x
