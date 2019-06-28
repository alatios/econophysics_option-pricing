LIBS=-L/usr/local/cuda/lib64 -lcurand
FLAGS=-std=c++11 -Wno-deprecated-gpu-targets
TARGS=rng.o main.o
NVCC=nvcc

ECHO=/bin/echo

all: $(TARGS)
	$(NVCC) $(FLAGS) $(LIBS) $(TARGS) -o main.x

%.o: %.cu
	$(NVCC) $(FLAGS) $(LIBS) -dc $< -o $@

clean:
	@rm -f *.x *.o && echo "Done cleaning."
