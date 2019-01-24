
CUDAFLAGS += -lineinfo -arch=sm_60 -rdc=true
CCFLAG += -lstdc++
CFLAGS += -D_FORCE_INLINES -c -w -O3
LDFLAGS += -lcuda -lcudart
CCFLAG += -lstdc++
INC_DIRS += -Iinclude

main:	build/main.o build/dust.o build/emissivity.o build/photon.o build/opacities.o build/grid.o build/frequencies.o build/global_functions.o build/stars.o
	@ echo "Linking MCRTGPU"
	@ mkdir -p bin
	@ nvcc build/*.o -o bin/mcrtgpu $(CUDAFLAGS) #$(CCFLAG) $(CFLAGS) $(LDFLAGS) $(CCFLAG)
	@ echo "The compilation has been completed successfully"

build/main.o: src/main.cu
	@ echo "Building Main"
	@ mkdir -p build
	@ nvcc $(CUDAFLAGS) $(INC_DIRS) src/main.cu -o build/main.o $(CCFLAG) $(CFLAGS) $(LDFLAGS) $(CCFLAG)

build/dust.o: src/dust.cu
	@ echo "Building Dust"
	@ mkdir -p build
	@ nvcc $(CUDAFLAGS) $(INC_DIRS) src/dust.cu -o build/dust.o $(CCFLAG) $(CFLAGS) $(LDFLAGS) $(CCFLAG)

build/emissivity.o: src/emissivity.cu
	@ echo "Building Emissivity"
	@ mkdir -p build
	@ nvcc $(CUDAFLAGS) $(INC_DIRS) src/emissivity.cu -o build/emissivity.o $(CCFLAG) $(CFLAGS) $(LDFLAGS) $(CCFLAG)

build/photon.o: src/photon.cu
	@ echo "Building Photon"
	@ mkdir -p build
	@ nvcc $(CUDAFLAGS) $(INC_DIRS) src/photon.cu -o build/photon.o $(CCFLAG) $(CFLAGS) $(LDFLAGS) $(CCFLAG)

build/opacities.o: src/opacities.cu
	@ echo "Building Opacities"
	@ mkdir -p build
	@ nvcc $(CUDAFLAGS) $(INC_DIRS) src/opacities.cu -o build/opacities.o $(CCFLAG) $(CFLAGS) $(LDFLAGS) $(CCFLAG)

build/grid.o: src/grid.cu
	@ echo "Building grid"
	@ mkdir -p build
	@ nvcc $(CUDAFLAGS) $(INC_DIRS) src/grid.cu -o build/grid.o $(CCFLAG) $(CFLAGS) $(LDFLAGS) $(CCFLAG)

build/frequencies.o: src/frequencies.cu
	@ echo "Building frequencies"
	@ mkdir -p build
	@ nvcc $(CUDAFLAGS) $(INC_DIRS) src/frequencies.cu -o build/frequencies.o $(CCFLAG) $(CFLAGS) $(LDFLAGS) $(CCFLAG)

build/global_functions.o: src/global_functions.cu
	@ echo "Building Global functions"
	@ mkdir -p build
	@ nvcc $(CUDAFLAGS) $(INC_DIRS) src/global_functions.cu -o build/global_functions.o $(CCFLAG) $(CFLAGS) $(LDFLAGS) $(CCFLAG)

build/stars.o: src/stars.cu
	@ echo "Building Stars"
	@ mkdir -p build
	@ nvcc $(CUDAFLAGS) $(INC_DIRS) src/stars.cu -o build/stars.o $(CCFLAG) $(CFLAGS) $(LDFLAGS) $(CCFLAG)

clean:
	@ echo "Cleaning MCRTGPU folders.."
	@ rm -rf build/*
	@ rm -rf bin/*
