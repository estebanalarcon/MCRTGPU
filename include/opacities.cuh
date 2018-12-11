#ifndef OPACITIES_CUH
#define OPACITIES_CUH

#include "structs.cuh"
DustOpacity* allocateMemoryToDustOpacity(int iFormat, int numSpec);
void allocateMemoryToKappas(DustOpacity* dustOpacity, int numLambdas, int iSpec);
void deallocateDustOpacities(DustOpacity* dustOpacity);
DustOpacity* dustOpacityTransferToDevice(DustOpacity* h_dustOpacity, int numFreq);
void checkOpacities(DustOpacity* dustOpacity, int iSpec, char* nameFile, int numLambdas);
void convertLambdasToFrequency(DustOpacity* dustOpacity);
void remapOpacitiesValues(DustOpacity* dustOpacity, FrequenciesData* freqData);
void readDustKappa(char* nameDust, DustOpacity* dustOpacity, int iSpec);
DustOpacity* readDustOpacity();
DustOpacity* setUpDustOpacity(FrequenciesData* freqData);
#endif
