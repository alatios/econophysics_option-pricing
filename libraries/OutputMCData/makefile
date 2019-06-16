LIBS=
#-lcuda -lm -L/usr/local/cuda/lib/ -lcudart
FLAGS=-std=c++11 -Wno-deprecated-gpu-targets
TARGS=UnitTest.o ../InputMarketData/Input_market_data.o ../InputOptionData/Input_option_data.o Output_MC_data.o
NVCC=nvcc

ECHO=/bin/echo

all: $(TARGS)
	$(NVCC) $(FLAGS) $(LIBS) $(TARGS) -o UnitTest.x

obj: %.o

%.o: %.cu
	$(NVCC) $(FLAGS) $(LIBS) -dc $< -o $@

clean:
	rm -f *.x *.o

run:
	cd ../InputMarketData  make obj && cd ../InputOptionData  make obj && cd ../OutputMCData && ./clean.sh && make && ./UnitTest.x
