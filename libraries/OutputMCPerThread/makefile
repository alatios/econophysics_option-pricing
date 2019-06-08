LIBS=
#-lcuda -lm -L/usr/local/cuda/lib/ -lcudart
FLAGS=-std=c++11 -Wno-deprecated-gpu-targets
TARGS=Output_MC_per_thread.o UnitTest.o
NVCC=nvcc

ECHO=/bin/echo

all: build

build: $(TARGS)
        $(NVCC) $(FLAGS) $(LIBS) $(TARGS) -o UnitTest.x

obj: %.o

%.o: %.cu
        $(NVCC) $(FLAGS) $(LIBS) -dc $< -o $@

clean:
        @rm -f *.x *.o && echo "Done cleaning."

run:
        @chmod u+x clean.sh
        @./clean.sh
        @make --no-print-directory
        @$(ECHO) "Running Output_MC_per_thread unit test..."
        @./UnitTest.x
