LIBS=
#-lcuda -lm -L/usr/local/cuda/lib/ -lcudart
FLAGS=-std=c++11 -Wno-deprecated-gpu-targets
TARGS=../InputGPUData/Input_gpu_data.o Input_MC_data.o UnitTest.o
NVCC=nvcc

ECHO=/bin/echo

all: $(TARGS)
	$(NVCC) $(FLAGS) $(LIBS) $(TARGS) -o UnitTest.x

obj: %.o

%.o: %.cu
	$(NVCC) $(FLAGS) $(LIBS) -dc $< -o $@

clean:
	rm -f *.x *.o

esegui:
	./clean.sh && make && ./UnitTest.x
