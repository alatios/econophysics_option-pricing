LIBS=
#-lcuda -lm -L/usr/local/cuda/lib/ -lcudart
FLAGS=-std=c++11 -Wno-deprecated-gpu-targets
TARGS=UnitTest.o ../InputOptionData/Input_option_data.o ../InputMarketData/Input_market_data.o Path.o
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
	cd ../InputMarketData  make obj && cd ../InputOptionData  make obj && cd ../Path && ./clean.sh && make && ./UnitTest.x
