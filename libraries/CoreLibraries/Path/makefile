FLAGS=-std=c++11 -Wno-deprecated-gpu-targets
TARGS=UnitTest.o ../InputOptionData/Input_option_data.o ../InputMarketData/Input_market_data.o Path.o
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
	@$(ECHO) "Running Path unit test..."
	@./UnitTest.x
