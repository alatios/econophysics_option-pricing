FLAGS= -std=c++11 -Wno-deprecated-gpu-targets
TARGS= \
UnitTest.o \
../InputMarketData/Input_market_data.o \
../InputOptionData/Input_option_data.o \
Output_MC_data.o

NVCC=nvcc

ECHO=/bin/echo

all: $(TARGS)
	$(NVCC) $(FLAGS) $(TARGS) -o UnitTest.x

%.o: %.cu
	$(NVCC) $(FLAGS) -dc $< -o $@

clean:
	@rm -f *.x *.o && echo "Done cleaning."

run: clean
	@make --no-print-directory
	@$(ECHO) "Running Output_MC_data unit test..."
	@./UnitTest.x
