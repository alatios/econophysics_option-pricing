LIBS=
#-lcuda -lm -L/usr/local/cuda/lib/ -lcudart
FLAGS=-std=c++11 -Wno-deprecated-gpu-targets
TARGS= ../InputMarketData/Input_market_data.o ../Path/Path.o ../InputOptionData/Input_option_data.o Path_per_thread.o UnitTest.o
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
	cd ../Path  make obj && cd ../PathPerThread && ./clean.sh &&  make && ./UnitTest.x
