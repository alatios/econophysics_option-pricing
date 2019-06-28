FLAGS=-std=c++11 -Wno-deprecated-gpu-targets
TARGS=main.o rng.o correlationtest.o
NVCC=nvcc

ECHO=/bin/echo

all: $(TARGS)
	$(NVCC) $(FLAGS) $(TARGS) -o main.x

%.o: %.cu
	$(NVCC) $(FLAGS) -dc $< -o $@

clean:
	@rm -f *.x *.o && echo "Done cleaning."
