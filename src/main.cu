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

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

int main(void){
  //cudaDeviceReset();
  int dev = 0;
  cudaDeviceProp deviceProp;
  gpuErrchk(cudaGetDeviceProperties(&deviceProp, dev));
  printf("Using Device %d: %s\n", dev, deviceProp.name);
  // set up device

  cudaSetDevice(dev);
  float nTemp = 1000.0;
  float temp0 = 0.01;
  float temp1 = 100000.0;
  int numPhotons = 1024*400;
  int blockSize = 256;
  int maxParallelPhotons = 1024*400;
  int numStreams = numPhotons/maxParallelPhotons;

  //read, process input data and transfer to device
  Grid* grid = setUpGrid();
  Grid* d_grid = gridTransferToDevice(grid);

  FrequenciesData* freqData = setUpFrequenciesData();
  FrequenciesData* d_freqData = frequenciesTransferToDevice(freqData);

  Stars* stars = setUpStars(freqData,grid, numPhotons);
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
  //int nx = grid->nCoord[0];
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

  Photon* photons = allocatePhotons(maxParallelPhotons, numSpec, numFrequencies);
  Photon* d_photons = photonsTransferToDevice(photons, maxParallelPhotons, numSpec, numFrequencies);
  deallocatePhotons(photons, maxParallelPhotons);
  printf("End transfers...\n");

  for (int i=0 ; i<numStreams ; i++){
    inicializeInicialPhoton<<<maxParallelPhotons/blockSize,blockSize>>>(d_photons, d_freqData, d_stars, d_grid, d_dustDensity, d_dustOpacity);
    gpuErrchk(cudaDeviceSynchronize());
    printf("End inicializeInicialPhoton\n");
    //<<<maxParallelPhotons/blockSize,blockSize>>>
    launchPhotons<<<maxParallelPhotons/blockSize,blockSize>>>(d_photons, d_freqData, d_stars,d_grid,d_dustDensity,d_dustOpacity, d_emissivityDb, d_dustTemperature);
    gpuErrchk(cudaDeviceSynchronize());
    //printf("End launchPhotons \n");
    printf("End stream%d\n\n",i);
  }
  printf("Convert energy to temperature...\n");
  int totalPositions = dustTemperature->totalPositions;
  printf("Total cells: %d\n",totalPositions);

  int numBlocks = totalPositions / blockSize;
  if (totalPositions > blockSize*numBlocks){
    numBlocks++;
  }
  printf("numBlocks for temp=%d \n",numBlocks);
  convertEnergyToTemperature<<<numBlocks,blockSize>>>(d_dustTemperature,d_dustDensity,d_grid,d_emissivityDb);
  gpuErrchk(cudaDeviceSynchronize());


  writeDustTemperature(dustTemperature, d_dustTemperature, dustDensity);
  //deallocateDustTemperature(dustTemperature, numSpec, ny,nz);
  cudaDeviceReset();
}