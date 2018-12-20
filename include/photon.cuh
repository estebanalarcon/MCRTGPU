#ifndef PHOTON_CUH
#define PHOTON_CUH
#include "structs.cuh"
#include <stdio.h>
#include <stdlib.h>
__global__ void kernelWalkNextEvent(Photon* d_photons, Stars* d_stars, DustDensity* d_dustDensity, DustOpacity* d_dustOpacity,
  Grid* d_grid, DustTemperature* d_dustTemperature);
__host__ bool arePhotonsOnGrid(bool* d_onGrid, bool* h_onGrid, int numPhotons);
__host__ void transferOnGridArrayToHost(bool* d_photonsOnGrid, bool* h_photonsOnGrid, int numPhotons);
__global__ void getPhotonsOnGrid(Photon* d_photons, bool* d_photonsOnGrid);
__global__ void kernelDoScatteringEvent(Photon* photons);
__global__ void kernelDoAbsorptionEvent(Photon* photons, Grid* grid, FrequenciesData* freqData, Stars* stars,
  DustDensity* dustDensity, DustOpacity* dustOpacity, EmissivityDatabase* emissivityDb, DustTemperature* dustTemperature);
__device__ void divideAbsorvedEnergy(Photon* photon, Stars* stars, DustDensity* dustDensity, DustOpacity* dustOpacity);
__device__ void addTemperatureDecoupled(Photon* photon, DustDensity* dustDensity,Grid* grid, EmissivityDatabase* emissivityDb, DustTemperature* dustTemperature);
__device__ void doAbsorptionEvent(Photon* photon, Grid* grid, FrequenciesData* freqData, Stars* stars,
  DustDensity* dustDensity, DustOpacity* dustOpacity, EmissivityDatabase* emissivityDb, DustTemperature* dustTemperature);
__device__ double advanceToNextPosition(Photon* photon, Grid* grid);
__device__ bool photonIsOnGrid(Photon* photon, Grid* grid);
__device__ void getTaupath(Photon* photon, curandState* state);
__device__ void findNewFrequencyInu(Photon* photon, curandState* state, double* specCum, FrequenciesData* freqData);
__device__ void checkUnitVector(double* direction);
__device__ void getRandomDirectionSimple(Photon* photon, curandState *state);
__device__ void findStar(Photon* photon, Stars* d_stars, curandState* state);
__host__ void setUpPhoton(Photon* photon, int numSpec, int numFreq);
__host__ void freePhoton(Photon* photon);
__device__ void calculateOpacityCoefficients(double minorDistance, Photon* photon, DustDensity* dustDensity,
  DustOpacity* dustOpacity);
__device__ void walkNextEvent(Photon* d_photons, Stars* d_stars, DustDensity* d_dustDensity, DustOpacity* d_dustOpacity,
  Grid* d_grid, DustTemperature* d_dustTemperature,  double cellWallsX[], double cellWallsY[], double cellWallsZ[]);
__device__ void inicializePositionPhoton(Photon* photon, Stars* d_stars, Grid* d_grid);
__host__ Photon* allocatePhotons(int numPhotons, int numSpec, int numFreq);
__host__ void deallocatePhotons(Photon* photons, int numPhotons);
__host__ Photon* photonsTransferToDevice(Photon* h_photons, int numPhotons, int numSpec, int numFreq);
__global__ void inicializeInicialPhoton(Photon* d_photons, FrequenciesData* d_freqData, Stars* d_stars, Grid* d_grid,
  DustDensity* d_dustDensity, DustOpacity* d_dustOpacity);
__global__ void launchPhotons(Photon* d_photons, FrequenciesData* d_freqData, Stars* d_stars, Grid* d_grid,
  DustDensity* d_dustDensity, DustOpacity* d_dustOpacity, //DustTemperature* d_dustTemperature,
  EmissivityDatabase* d_emissivityDb, DustTemperature* d_dustTemperature);
  __device__ double advanceToNextPositionTest(Photon* photon, Grid* grid, double cellWallsX[], double cellWallsY[], double cellWallsZ[]);
  __global__ void testId();
  __global__ void testId2();

#endif
