#ifndef DUST_CUH
#define DUST_CUH

#include "structs.cuh"

__global__ void printEner(DustTemperature* dustTemperature, DustDensity* dustDensity);
__global__ void printTemp(DustTemperature* dustTemperature, DustDensity* dustDensity);
__host__ DustTemperature* allocateMemoryToDustTemperature(int numSpec, Grid* grid);
__host__ void deallocateCumulEner(DustTemperature* dustTemperature, int numSpec, int ny, int nz);
__host__ void deallocateDustTemperature(DustTemperature* dustTemperature, int numSpec, int ny, int nz);
__host__ void inicializeDustTemperature(DustTemperature* dustTemperature, DustDensity* dustDensity);
__host__ DustDensity* allocateMemoryToDustDensity(int iFormat, int numCells, int numSpec, Grid* grid);
__host__ void deallocateDensities(DustDensity* dustDensity);
__host__ DustDensity* dustDensityTransferToDevice(DustDensity* h_dustDensity);
__host__ void writeDustTemperature(DustTemperature* h_dustTemperature, DustTemperature* d_dustTemperature, DustDensity* h_dustDensity);
__host__ DustTemperature* dustTemperatureTransferToDevice(DustTemperature* h_dustTemperature, DustDensity* h_dustDensity);
__host__ DustDensity* setUpDustDensity(Grid* grid);
__host__ DustTemperature* setUpDustTemperature(DustDensity* dustDensity, Grid* grid);

#endif
