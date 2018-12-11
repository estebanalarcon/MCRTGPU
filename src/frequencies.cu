#include <math.h>
#include "constants.cuh"
#include "frequencies.cuh"


FrequenciesData* allocateMemoryToFrequenciesData(int numFrequencies){
  FrequenciesData* frequenciesData = (FrequenciesData*)malloc(sizeof(FrequenciesData));
  frequenciesData->numFrequencies = numFrequencies;
  frequenciesData->frequencies = (float*) malloc (sizeof(float)*numFrequencies);
  frequenciesData->frequenciesDnu = (float*) malloc (sizeof(float)*numFrequencies);
  return frequenciesData;
}

void deallocateFrequenciesData(FrequenciesData* freqData){
  free(freqData->frequencies);
  free(freqData->frequenciesDnu);
  free(freqData);
}

__host__ FrequenciesData* readWavelengthMicron(){
  FrequenciesData* freqData;
  int numFrequencies;
  char line[30];
  const char* nameFile = "inputs/wavelength_micron.inp";
  FILE* wavelengthMicron = fopen(nameFile,"r");
  if (wavelengthMicron == NULL){
    printf("Failed to open wavelength_micron. Maybe don't exist\n");
    exit(1);
  }else{
    fgets(line, 30, wavelengthMicron);
    numFrequencies = atoi(line);
    if (numFrequencies < 1){
      printf("ERROR: File explicit < 1 point\n");
      exit(1);
    }else{
      printf("Number frequencies: %d\n",numFrequencies);
      freqData = allocateMemoryToFrequenciesData(numFrequencies);
      int count=0;
      while (fgets(line, 30, wavelengthMicron) != NULL){
        freqData-> frequencies[count] = atof(line);
        count++;
      }
      if (count != numFrequencies){
        printf("ERROR: number of frequencies listed is not the same\n");
        exit(1);
      }
      fclose(wavelengthMicron);
    }
  }
  return freqData;
}

__host__ void convertMicronToFrequency(FrequenciesData* freqData){
  int i;
  for (i=0; i < freqData->numFrequencies ; i++){
    //printf("freqData%d = %f\n",i,freqData->frequencies[i]);
    freqData->frequencies[i] = 10000 * C  / freqData->frequencies[i];
  }
}

__host__ void calculateFrequenciesDnu(FrequenciesData* freqData){
  int numFreq = freqData->numFrequencies;
  if (numFreq == 1){
    freqData->frequenciesDnu[0] = 1;
  }else{
    freqData->frequenciesDnu[0] = 0.5 * fabs(freqData->frequencies[1] - freqData->frequencies[0]);
    freqData->frequenciesDnu[numFreq-1] = 0.5 * fabs(freqData->frequencies[numFreq-1] - freqData->frequencies[numFreq-2]);

    int i;
    for (i=1 ; i < numFreq-1 ; i++){
      freqData->frequenciesDnu[i] = 0.5 * fabs(freqData->frequencies[i+1] - freqData->frequencies[i-1]);
    }
  }
}

FrequenciesData* frequenciesTransferToDevice(FrequenciesData* h_freqData){
  printf("Transfer frequenciesData to device...\n");
  FrequenciesData* d_freqData;
  float* frequencies;
  float* frequenciesDnu;
  size_t sizeFrequencies = sizeof(float) * (h_freqData->numFrequencies);

  cudaMalloc((void**)&(d_freqData), sizeof(FrequenciesData) );
  cudaMalloc((void**)&frequencies, sizeFrequencies);
  cudaMalloc((void**)&frequenciesDnu, sizeFrequencies);

  cudaMemcpy(d_freqData, h_freqData, sizeof(FrequenciesData), cudaMemcpyHostToDevice);
  cudaMemcpy(&(d_freqData->frequencies), &frequencies, sizeof(float *), cudaMemcpyHostToDevice);
  cudaMemcpy(&(d_freqData->frequenciesDnu), &frequenciesDnu, sizeof(float *), cudaMemcpyHostToDevice);

  cudaMemcpy(frequencies, h_freqData->frequencies, sizeFrequencies, cudaMemcpyHostToDevice);
  cudaMemcpy(frequenciesDnu, h_freqData->frequenciesDnu, sizeFrequencies, cudaMemcpyHostToDevice);

  return d_freqData;
}

FrequenciesData* setUpFrequenciesData(){
  printf("Set up frequencies...\n");
  FrequenciesData* freqData = readWavelengthMicron();
  convertMicronToFrequency(freqData);
  calculateFrequenciesDnu(freqData);
  return freqData;
}
