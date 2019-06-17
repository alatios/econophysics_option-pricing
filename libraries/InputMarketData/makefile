FLAGS=-std=c++11 -Wno-deprecated-gpu-targets
TARGS=UnitTest.o Input_market_data.o
NVCC=nvcc

ECHO=/bin/echo

all: build

build: $(TARGS)
	$(NVCC) $(FLAGS) $(TARGS) -o UnitTest.x

%.o: %.cu
	$(NVCC) $(FLAGS) -dc $< -o $@

clean:
	@rm -f *.x *.o && echo "Done cleaning."

run: clean
	@make --no-print-directory
	@$(ECHO) "Running Input_market_data unit test..."
	@./UnitTest.x
