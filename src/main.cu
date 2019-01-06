#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "photon.cuh"
#include "opacities.cuh"
#include "dust.cuh"
#include "stars.cuh"
#include "grid.cuh"
#include "frequencies.cuh"
#include "emissivity.cuh"
#include "global_functions.cuh"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
__global__ void kernelGetTaupath(Photon* d_photons){
  unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (d_photons[tid].onGrid){
    float rn = curand_uniform(&d_photons[tid].state);
    d_photons[tid].taupathTotal = - log(1.0-rn);
    d_photons[tid].taupathGone = 0.0;
  }
}

void countPhotonsInGrid(bool* h_onGrid, int numPhotons){
  int count=0;
  for (int i=0 ; i<numPhotons ; i++){
    if (h_onGrid[i]){
      count++;
    }
  }
  printf("countPhotons = %d\n",count);
}

void executePhotons(SimulationParameters param, Photon* d_photons,
  FrequenciesData* d_freqData, Stars* d_stars, Grid* d_grid, DustDensity* d_dustDensity,
  DustOpacity* d_dustOpacity, EmissivityDatabase* d_emissivityDb,  DustTemperature* d_dustTemperature, SimulationParameters* d_params){
    inicializeInicialPhoton<<<param.numParallelPhotons/param.blockSize,param.blockSize>>>(d_photons, d_freqData, d_stars, d_grid, d_dustDensity, d_dustOpacity);
    //inicializeInicialPhoton<<<1,1>>>(d_photons, d_freqData, d_stars, d_grid, d_dustDensity, d_dustOpacity);
    gpuErrchk(cudaDeviceSynchronize());
    launchPhotons<<<param.numParallelPhotons/param.blockSize,param.blockSize>>>(d_photons, d_freqData, d_stars,d_grid,d_dustDensity,d_dustOpacity, d_emissivityDb, d_dustTemperature,d_params);
    //launchPhotons<<<1,1>>>(d_photons, d_freqData, d_stars,d_grid,d_dustDensity,d_dustOpacity, d_emissivityDb, d_dustTemperature);
    gpuErrchk(cudaDeviceSynchronize());
  }

void executePhotons2(int blockSize, int numParallelPhotons,Photon* d_photons,
  FrequenciesData* d_freqData, Stars* d_stars, Grid* d_grid, DustDensity* d_dustDensity,
  DustOpacity* d_dustOpacity, EmissivityDatabase* d_emissivityDb,  DustTemperature* d_dustTemperature,
  bool* h_onGrid, bool* d_onGrid){
    inicializeInicialPhoton<<<numParallelPhotons/blockSize,blockSize>>>(d_photons, d_freqData, d_stars, d_grid, d_dustDensity, d_dustOpacity);
    gpuErrchk(cudaDeviceSynchronize());
    kernelWalkNextEvent<<<numParallelPhotons/blockSize,blockSize>>>(d_photons,d_stars,d_dustDensity,d_dustOpacity,d_grid,d_dustTemperature);
    gpuErrchk(cudaDeviceSynchronize());
    getPhotonsOnGrid<<<numParallelPhotons/blockSize,blockSize>>>(d_photons, d_onGrid);
    gpuErrchk(cudaDeviceSynchronize());
    while (arePhotonsOnGrid(d_onGrid, h_onGrid, numParallelPhotons)){
      //countPhotonsInGrid(h_onGrid, numParallelPhotons);
      //printf("arePhotonsOnGrid%d...\n",cont);
      kernelDoAbsorptionEvent<<<numParallelPhotons/blockSize,blockSize>>>(d_photons,d_grid,d_freqData,d_stars,d_dustDensity,d_dustOpacity,d_emissivityDb,d_dustTemperature);
      gpuErrchk(cudaDeviceSynchronize());
      //kernelDoScatteringEvent<<<numParallelPhotons/blockSize,blockSize>>>(d_photons);
      //gpuErrchk(cudaDeviceSynchronize());
      kernelGetTaupath<<<numParallelPhotons/blockSize,blockSize>>>(d_photons);
      gpuErrchk(cudaDeviceSynchronize());
      kernelWalkNextEvent<<<numParallelPhotons/blockSize,blockSize>>>(d_photons,d_stars,d_dustDensity,d_dustOpacity,d_grid,d_dustTemperature);
      gpuErrchk(cudaDeviceSynchronize());
      getPhotonsOnGrid<<<numParallelPhotons/blockSize,blockSize>>>(d_photons, d_onGrid);
      gpuErrchk(cudaDeviceSynchronize());

  }
  //printf("hola\n");
}

int main(int argc, char **argv){
  //cudaDeviceReset();
  int dev = 0;
  cudaDeviceProp deviceProp;
  gpuErrchk(cudaGetDeviceProperties(&deviceProp, dev));
  gpuErrchk(cudaSetDevice(dev));
  printf("Using Device %d: %s\n", dev, deviceProp.name);


  SimulationParameters param = readInputParameters(argc, argv);
  checkSimulationParameters(param);
  float nTemp = 1000.0;
  float temp0 = 0.01;
  float temp1 = 100000.0;
  //int numPhotons = 1024*10000;
  //int blockSize = 32;
  //int maxParallelPhotons = 1024*50;
  int numStreams = param.numPhotons/param.numParallelPhotons;
  printf("Number of streams = %d\n",numStreams);

  //bool* d_onGrid;
  //cudaMalloc((void**)&(d_onGrid), sizeof(bool)*param.maxParallelPhotons );

  //bool* h_onGrid =(bool*) malloc(sizeof(bool)*param.maxParallelPhotons);

  //read, process input data and transfer to device
  Grid* grid = setUpGrid();
  Grid* d_grid = gridTransferToDevice(grid);

  FrequenciesData* freqData = setUpFrequenciesData();
  FrequenciesData* d_freqData = frequenciesTransferToDevice(freqData);

  Stars* stars = setUpStars(freqData,grid, param.numPhotons);
  Stars* d_stars = starsTransferToDevice(stars, freqData->numFrequencies);

  DustDensity* dustDensity = setUpDustDensity(grid);
  DustDensity* d_dustDensity = dustDensityTransferToDevice(dustDensity);

  DustOpacity* dustOpacity = setUpDustOpacity(freqData);
  DustOpacity* d_dustOpacity = dustOpacityTransferToDevice(dustOpacity, freqData->numFrequencies);

  EmissivityDatabase* emissivityDb = setUpEmissivityDatabase(nTemp, temp0, temp1, dustOpacity, freqData);
  EmissivityDatabase* d_emissivityDb = emissivityDbTransferToDevice(emissivityDb, freqData->numFrequencies, dustDensity->numSpec);

  DustTemperature* dustTemperature = setUpDustTemperature(dustDensity, grid);
  DustTemperature* d_dustTemperature  = dustTemperatureTransferToDevice(dustTemperature, dustDensity);

  int numSpec = dustDensity->numSpec;
  int numFrequencies = freqData->numFrequencies;
  int nx = grid->nCoord[0];
  int ny = grid->nCoord[1];
  int nz = grid->nCoord[2];

  //free cpu memory
  deallocateCumulEner(dustTemperature, numSpec, ny,nz);
  deallocateEmissivityDatabase(emissivityDb, nTemp, numSpec);
  deallocateDustOpacities(dustOpacity);
  deallocateDensities(dustDensity);
  deallocateStars(stars);
  deallocateFrequenciesData(freqData);
  deallocateGrid(grid);

  SimulationParameters* d_params = parametersTransferToDevice(param);
  Photon* photons1 = allocatePhotons(param.numParallelPhotons, numSpec, numFrequencies);
  Photon* d_photons1 = photonsTransferToDevice(photons1, param.numParallelPhotons, numSpec, numFrequencies);
  deallocatePhotons(photons1, param.numParallelPhotons);

  printf("End transfers...\n");

  for (int i=0 ; i<numStreams ; i++){
    printf("Stream %d...\n",i);
    executePhotons(param, d_photons1,d_freqData, d_stars,d_grid,d_dustDensity,d_dustOpacity, d_emissivityDb, d_dustTemperature, d_params);
    //executePhotons2(blockSize, maxParallelPhotons,d_photons1,d_freqData, d_stars,d_grid,d_dustDensity,d_dustOpacity, d_emissivityDb, d_dustTemperature,h_onGrid,d_onGrid);
  }
  printf("Convert energy to temperature...\n");
  int totalPositions = dustTemperature->totalPositions;
  //printf("Total cells: %d\n",totalPositions);

  int numBlocks = totalPositions / param.blockSize;
  if (totalPositions > param.blockSize*numBlocks){
    numBlocks++;
  }
  //printf("numBlocks for temp=%d \n",numBlocks);
  convertEnergyToTemperature<<<numBlocks,param.blockSize>>>(d_dustTemperature,d_dustDensity,d_grid,d_emissivityDb);
  gpuErrchk(cudaDeviceSynchronize());
  //printEner<<<1,1>>>(d_dustTemperature,d_dustDensity);
  //gpuErrchk(cudaDeviceSynchronize());


  writeDustTemperature(dustTemperature, d_dustTemperature, numSpec,nz,ny,nx);
  //deallocateDustTemperature(dustTemperature, numSpec, ny,nz);*/
  printf("End simulation\n");
  cudaDeviceReset();
}
