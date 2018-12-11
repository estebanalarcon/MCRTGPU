#ifndef EMISSIVITY_CUH
#define EMISSIVITY_CUH
#include "structs.cuh"

__device__ int pickRandomFreqDb(EmissivityDatabase* emissivityDb, Photon* photon,
  int numSpec, int numFreq, double* tempLocal, double* enerPart);
__device__ double computeDusttempEnergyBd(EmissivityDatabase* emissivityDb, double energy, int iSpec);
EmissivityDatabase* allocateMemoryToEmissivityDatabase(float nTemp, float temp0, float temp1, int numSpec, int numFreq);
void deallocateEmissivityDatabase(EmissivityDatabase* emissivityDb, float nTemp, int numSpec);
EmissivityDatabase* emissivityDbTransferToDevice(EmissivityDatabase* h_emissivityDb, int numFreq, int numSpec);
void setUpDbTemp(EmissivityDatabase* emissivityDb);
double abservfunc(double temp, double* cellAlpha, FrequenciesData* freqData, double* fnuDiff);
void setUpDbEnerTemp(EmissivityDatabase* emissivityDb, DustOpacity* dustOpacity, FrequenciesData* freqData);
void setUpDbCumul(EmissivityDatabase* emissivityDb, FrequenciesData* freqData, int numSpec);
__global__ void convertEnergyToTemperature(DustTemperature* d_dustTemperature, DustDensity* d_dustDensity, Grid* d_grid, EmissivityDatabase* d_emissivityDb);
EmissivityDatabase* setUpEmissivityDatabase(float nTemp, float temp0, float temp1, DustOpacity* dustOpacity, FrequenciesData* freqData);

#endif
