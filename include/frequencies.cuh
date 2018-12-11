#ifndef FREQUENCIES_CUH
#define FREQUENCIES_CUH
#include "structs.cuh"

FrequenciesData* allocateMemoryToFrequenciesData(int numFrequencies);
void deallocateFrequenciesData(FrequenciesData* freqData);
__host__ FrequenciesData* readWavelengthMicron();
__host__ void convertMicronToFrequency(FrequenciesData* freqData);
FrequenciesData* frequenciesTransferToDevice(FrequenciesData* h_freqData);
FrequenciesData* setUpFrequenciesData();

#endif
