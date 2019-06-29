FLAGS=-std=c++11 -Wno-deprecated-gpu-targets
TARGS_OUTPUT=RNG.o OutputTest.o
TARGS_CORRELATION=RNG.o CorrelationToolbox.o CorrelationTest.o
NVCC=nvcc

ECHO=/bin/echo

all: output corr

output:  $(TARGS_OUTPUT)
	$(NVCC) $(FLAGS) $(TARGS_OUTPUT) -o OutputTest

corr:  $(TARGS_CORRELATION)
	$(NVCC) $(FLAGS) $(TARGS_CORRELATION) -o CorrelationTest

%.o: %.cu
	$(NVCC) $(FLAGS) -dc $< -o $@

clean:
	@rm -f *.x *.o && echo "Done cleaning."
