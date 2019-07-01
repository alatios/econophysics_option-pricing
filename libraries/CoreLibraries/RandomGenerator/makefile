FLAGS=-std=c++11 -Wno-deprecated-gpu-targets
TARGS_OUTPUT=RNG.o OutputTest.o
TARGS_CORRELATION=RNG.o CorrelationToolbox.o CorrelationTest.o
NVCC=nvcc

ECHO=/bin/echo

all: output corr

output:  $(TARGS_OUTPUT)
	$(NVCC) $(FLAGS) $(TARGS_OUTPUT) -o OutputTest.x

corr:  $(TARGS_CORRELATION)
	$(NVCC) $(FLAGS) $(TARGS_CORRELATION) -o CorrelationTest.x

%.o: %.cu
	$(NVCC) $(FLAGS) -dc $< -o $@

clean:
	@rm -f *.x *.o && echo "Done cleaning."


run: clean
	@make --no-print-directory
	@$(ECHO) "Running RandomGenerator output test..."
	@./OutputTest.x
	@$(ECHO) "Running RandomGenerator correlation test..."
	@./CorrelationTest.x

