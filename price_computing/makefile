LIBS=
FLAGS=-std=c++11 -Wno-deprecated-gpu-targets
TARGS=../unit_test_2019_05_28/InputGPUData/Input_gpu_data.o ../unit_test_2019_05_28/InputMarketData/Input_market_data.o ../unit_test_2019_05_28/InputMCData/Input_MC_data.o ../unit_test_2019_05_28/InputOptionData/Input_option_data.o ../unit_test_2019_05_28/OutputMCData/Output_MC_data.o ../unit_test_2019_05_28/Path/Path.o ../unit_test_2019_05_28/PathPerThread/Path_per_thread.o ../RNG/rng.o main.o
NVCC=nvcc

ECHO=/bin/echo

all: $(TARGS)
	$(NVCC) $(FLAGS) $(LIBS) $(TARGS) -o main.x

obj: %.o

%.o: %.cu
	$(NVCC) $(FLAGS) $(LIBS) -dc $< -o $@

clean:
	cd ../unit_test_2019_05_28/InputGPUData && rm -f *.x *.o
	cd ../unit_test_2019_05_28/InputMarketData && rm -f *.x *.o
	cd ../unit_test_2019_05_28/InputMCData && rm -f *.x *.o
	cd ../unit_test_2019_05_28/InputOptionData && rm -f *.x *.o
	cd ../unit_test_2019_05_28/OutputMCData && rm -f *.x *.o
	cd ../unit_test_2019_05_28/Path && rm -f *.x *.o
	cd ../unit_test_2019_05_28/PathPerThread && rm -f *.x *.o
	cd ../RNG && rm -f *.x *.o
	rm -f *.x *.o
	echo "Done cleaning."
#esegui:
#	cd ../Path  make obj && cd ../PathPerThread && ./clean.sh &&  make && ./UnitTest.x
