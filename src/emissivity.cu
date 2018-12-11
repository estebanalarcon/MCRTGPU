#include <math.h>
#include "constants.cuh"
#include "emissivity.cuh"
#include "global_functions.cuh"

__device__ int pickRandomFreqDb(EmissivityDatabase* emissivityDb, Photon* photon,
  int numSpec, int numFreq, double* tempLocal, double* enerPart){
    //float sumTemp1, sumTemp2;
    int numCumul = numFreq+1;
    int intplt = 1;
    curandState state;
    double rn;
    int iSpec = 0;
    int iTemp = 0;
    int inuPick = 0;
    //double* enerCum = (double*)malloc(sizeof(double)*(numSpec+1));
    photon->enerCum[0] = 0;
    for (int i=1 ; i<(numSpec+1) ; i++){
      photon->enerCum[i] = photon->enerCum[i-1] + enerPart[i-1];
    }
    if (numSpec>1){
      rn = curand_uniform_double(&state);
      huntDouble(photon->enerCum, numSpec+1, rn, &iSpec);
      if ((iSpec<0 || (iSpec>numSpec-1))){
        printf("exit\n");
        //exit(1);
      }
    }
    //free(enerCum);
    //printf("iSpec=%d\n",*iSpec);
    //find temperature in the database
    //printf("tempLocal=%lg\n",tempLocal[*iSpec]);
    huntDouble(emissivityDb->dbTemp, emissivityDb->nTemp, tempLocal[iSpec], &iTemp);
    //printf("itemp hunted=%d\n",*iTemp);
    float eps = 0.0;

    if (iTemp >= emissivityDb->nTemp-1){
      printf("ERROR: Too high temperature discovered\n");
    }
    if (iTemp <= -1){
      iTemp = 0;
    }else{
      eps = (tempLocal[iSpec]-emissivityDb->dbTemp[iTemp]) /
        (emissivityDb->dbTemp[iTemp+1]-emissivityDb->dbTemp[iTemp]);
      if ((eps > 1) || (eps < 0)){
        printf("ERROR: in pick randomFreq, eps<0 or eps>1\n");
        printf("exit\n");
        //exit(1);
      }
    }

    if (intplt == 1){
      //printf("numFrequencies=%d\n",numFreq);
      //printf("itemp=%d\n",*iTemp);
      //float* dbCumul = (float*)malloc(sizeof(float)*numCumul);
      for (int inu=0 ; inu<numCumul ; inu++){
        //sumTemp1 = (1.0-eps)*emissivityDb->dbCumulNorm[iSpec][iTemp][inu];
        //sumTemp2 = eps*emissivityDb->dbCumulNorm[iSpec][iTemp+1][inu];
        //photon->dbCumul[inu] = sumTemp1+sumTemp2;
        photon->dbCumul[inu] = (1.0-eps)*emissivityDb->dbCumulNorm[iSpec][iTemp][inu] + eps*emissivityDb->dbCumulNorm[iSpec][iTemp+1][inu];
      }
      double rn = curand_uniform_double(&state);
      //rn = 0.20160651049169737;
      huntFloat(photon->dbCumul, numFreq, rn, &inuPick);
      //free(dbCumul);
    }else{
      //Now, within this species/size, find the frequency
      if (eps>0.5){
        if (iTemp < emissivityDb->nTemp-2){
          iTemp++;
        }
      }
      double rn = curand_uniform_double(&state);
      //verify dbCumulNorm
      huntFloat(emissivityDb->dbCumulNorm[iSpec][iTemp-1], numFreq, rn, &inuPick);
    }
    return inuPick;
  }

__device__ double computeDusttempEnergyBd(EmissivityDatabase* emissivityDb, double energy, int iSpec){
  //printf("in computeDusttempEnergyBd\n");
  double tempReturn=0;
  int itemp = 0;
  double logEner = log(energy);
  double eps;
  //printf("logEner=%lf\n", logEner);
  //printf("before, itemp=%d\n", *itemp);
  huntDouble(emissivityDb->dbLogEnerTemp[iSpec], emissivityDb->nTemp, logEner, &itemp);
  //printf("itemp=%d\n", *itemp);
  if (itemp >= emissivityDb->nTemp-1){
    printf("ERROR: Too high temperature discovered\n");
    printf("exit\n");
    //exit(1);

  }

  if (itemp <= -1){
    //Temperature presumably below lowest temp in dbase
    eps = energy/emissivityDb->dbEnerTemp[iSpec][0];
    if (eps >= 1){
      printf("exit\n");
      //exit(1);
    }
    tempReturn = eps * emissivityDb->dbTemp[0];
  }else{
    eps = (energy-emissivityDb->dbEnerTemp[iSpec][itemp]) /
      (emissivityDb->dbEnerTemp[iSpec][itemp+1]-emissivityDb->dbEnerTemp[iSpec][itemp]);
    if ((eps>1) || (eps<0)){
      printf("exit\n");
      //exit(1);
    }
    tempReturn = (1.0-eps)*emissivityDb->dbTemp[itemp]+eps*emissivityDb->dbTemp[itemp+1];
  }
  return tempReturn;
}

EmissivityDatabase* allocateMemoryToEmissivityDatabase(float nTemp, float temp0, float temp1, int numSpec, int numFreq){
  EmissivityDatabase* emissivityDb = (EmissivityDatabase*)malloc(sizeof(EmissivityDatabase));
  emissivityDb->nTemp = nTemp;
  emissivityDb->temp0 = temp0;
  emissivityDb->temp1 = temp1;
  emissivityDb->dbTemp = (double*)malloc(sizeof(double)*nTemp);
  emissivityDb->dbEnerTemp = (double**)malloc(sizeof(double*)*numSpec);
  emissivityDb->dbLogEnerTemp = (double**)malloc(sizeof(double*)*numSpec);
  emissivityDb->dbEmiss = (double***)malloc(sizeof(double**)*numSpec);
  emissivityDb->dbCumulNorm = (float***)malloc(sizeof(float**)*numSpec);



  for (int i=0 ; i<numSpec ; i++){
    emissivityDb->dbEnerTemp[i] = (double*)malloc(sizeof(double)*nTemp);
    emissivityDb->dbLogEnerTemp[i] = (double*)malloc(sizeof(double)*nTemp);
    emissivityDb->dbEmiss[i] = (double**)malloc(sizeof(double*)*nTemp);
    emissivityDb->dbCumulNorm[i] = (float**)malloc(sizeof(float*)*nTemp);

    for (int j=0 ; j<nTemp ; j++){
      emissivityDb->dbEmiss[i][j] = (double*)malloc(sizeof(double)*numFreq);
      emissivityDb->dbCumulNorm[i][j] = (float*)malloc(sizeof(float)*(numFreq+1));

    }
  }
  return emissivityDb;
}

void deallocateEmissivityDatabase(EmissivityDatabase* emissivityDb, float nTemp, int numSpec){
  free(emissivityDb->dbTemp);
  for (int i=0 ; i<numSpec ; i++){
      free(emissivityDb->dbLogEnerTemp[i]);
      free(emissivityDb->dbEnerTemp[i]);
      for (int j=0 ; j<nTemp ; j++){
        free(emissivityDb->dbEmiss[i][j]);
        free(emissivityDb->dbCumulNorm[i][j]);
      }
      free(emissivityDb->dbEmiss[i]);
      free(emissivityDb->dbCumulNorm[i]);
  }
  free(emissivityDb->dbLogEnerTemp);
  free(emissivityDb->dbEnerTemp);
  free(emissivityDb->dbEmiss);
  free(emissivityDb->dbCumulNorm);
  free(emissivityDb);
}


EmissivityDatabase* emissivityDbTransferToDevice(EmissivityDatabase* h_emissivityDb, int numFreq, int numSpec){
  printf("Transfer EmissivityDatabase to device...\n");
    EmissivityDatabase* d_emissivityDb;
    float* dbTemp;
    double** dbEnerTemp;
    double** dbLogEnerTemp;
    double*** dbEmiss;
    float*** dbCumulNorm;

    double** supDbEnerTemp;
    double** supDbLogEnerTemp;

    double*** support3emiss;
    float*** support3cumul;
    double** support2emiss;
    float**support2cumul;

    int nTemp = (int) h_emissivityDb->nTemp;
    //d_emissivityDb
    cudaMalloc((void **) &d_emissivityDb,sizeof(EmissivityDatabase));
    cudaMemcpy(d_emissivityDb, h_emissivityDb, sizeof(EmissivityDatabase), cudaMemcpyHostToDevice);

    //dbTemp
    cudaMalloc((void**)&dbTemp, sizeof(double) * nTemp);
    cudaMemcpy(&(d_emissivityDb->dbTemp), &dbTemp, sizeof(double *), cudaMemcpyHostToDevice);
    cudaMemcpy(dbTemp, h_emissivityDb->dbTemp, sizeof(double) * nTemp, cudaMemcpyHostToDevice);

    //dbEnerTemp,dbLogEnerTemp
    cudaMalloc((void **) &dbEnerTemp,numSpec*sizeof(double*));
    cudaMalloc((void **) &dbLogEnerTemp,numSpec*sizeof(double*));
    supDbEnerTemp =(double**) malloc(sizeof(double*)*numSpec);
    supDbLogEnerTemp =(double**) malloc(sizeof(double*)*numSpec);
    for (int i=0;i<numSpec;i++){
      cudaMalloc((void**) &supDbEnerTemp[i],sizeof(double)*nTemp);
      cudaMalloc((void**) &supDbLogEnerTemp[i],sizeof(double)*nTemp);
      cudaMemcpy(supDbEnerTemp[i],h_emissivityDb->dbEnerTemp[i],nTemp*sizeof(double),cudaMemcpyHostToDevice);
      cudaMemcpy(supDbLogEnerTemp[i],h_emissivityDb->dbLogEnerTemp[i],nTemp*sizeof(double),cudaMemcpyHostToDevice);
    }
    cudaMemcpy(dbEnerTemp,supDbEnerTemp,sizeof(double*)*numSpec,cudaMemcpyHostToDevice);
    cudaMemcpy(dbLogEnerTemp,supDbLogEnerTemp,sizeof(double*)*numSpec,cudaMemcpyHostToDevice);

    cudaMemcpy(&(d_emissivityDb->dbEnerTemp), &dbEnerTemp, sizeof(double**), cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_emissivityDb->dbLogEnerTemp), &dbLogEnerTemp, sizeof(double**), cudaMemcpyHostToDevice);
    free(supDbEnerTemp);
    free(supDbLogEnerTemp);

    //dbEmiss, dbCumulNorm
    cudaMalloc((void **) &dbEmiss,numSpec*sizeof(double**));
    cudaMalloc((void **) &dbCumulNorm,numSpec*sizeof(float**));
    support3emiss =(double***) malloc(sizeof(double**)*numSpec);
    support3cumul =(float***) malloc(sizeof(float**)*numSpec);
    for (int i=0;i<numSpec;i++){
      cudaMalloc((void**) &support3emiss[i],sizeof(double*)*nTemp);
      cudaMalloc((void**) &support3cumul[i],sizeof(float*)*nTemp);
      support2emiss =(double**) malloc(sizeof(double*)*nTemp);
      support2cumul =(float**) malloc(sizeof(float*)*nTemp);
      for (int j=0;j<nTemp;j++){
        cudaMalloc((void**) &support2emiss[j],sizeof(double)*numFreq);
        cudaMalloc((void**) &support2cumul[j],sizeof(float)*(numFreq+1));
        cudaMemcpy(support2emiss[j],h_emissivityDb->dbEmiss[i][j],numFreq*sizeof(double),cudaMemcpyHostToDevice);
        cudaMemcpy(support2cumul[j],h_emissivityDb->dbCumulNorm[i][j],(numFreq+1)*sizeof(float),cudaMemcpyHostToDevice);
      }
      cudaMemcpy(support3emiss[i],support2emiss,sizeof(double*)*nTemp,cudaMemcpyHostToDevice);
      cudaMemcpy(support3cumul[i],support2cumul,sizeof(float*)*nTemp,cudaMemcpyHostToDevice);
      free(support2cumul);
      free(support2emiss);
    }
    cudaMemcpy(dbEmiss,support3emiss,sizeof(double**)*numSpec,cudaMemcpyHostToDevice);
    cudaMemcpy(dbCumulNorm,support3cumul,sizeof(float**)*numSpec,cudaMemcpyHostToDevice);
    free(support3cumul);
    free(support3emiss);

    cudaMemcpy(&(d_emissivityDb->dbEmiss), &dbEmiss, sizeof(double***), cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_emissivityDb->dbCumulNorm), &dbCumulNorm, sizeof(float***), cudaMemcpyHostToDevice);

    return d_emissivityDb;
}

void setUpDbTemp(EmissivityDatabase* emissivityDb){
  int i;
  float x,y;
  for (i=0 ; i<emissivityDb->nTemp ; i++){
    x = emissivityDb->temp1/emissivityDb->temp0;
    y = i/(emissivityDb->nTemp-1.0);
    emissivityDb->dbTemp[i] = emissivityDb->temp0 * pow(x,y);
  }
}

double abservfunc(double temp, double* cellAlpha, FrequenciesData* freqData, double* fnuDiff){
  double demis = 0;
  int i;
  for (i=0 ; i<freqData->numFrequencies; i++){
    fnuDiff[i] = cellAlpha[i] * blackbodyPlanck(temp, freqData->frequencies[i]);
    demis += fnuDiff[i] * freqData->frequenciesDnu[i];
  }
  demis = demis * 4 * PI;
  return demis;
}

void setUpDbEnerTemp(EmissivityDatabase* emissivityDb, DustOpacity* dustOpacity, FrequenciesData* freqData){
  int iSpec, iLam, iTemp;
  double* cellAlpha;
  double demis;
  for (iSpec = 0 ; iSpec<dustOpacity->numSpec ; iSpec++){
    cellAlpha = (double*)malloc(sizeof(double)*freqData->numFrequencies);

    for (iLam=0 ; iLam<freqData->numFrequencies ; iLam++){
      cellAlpha[iLam] = dustOpacity->kappaA[iSpec][iLam];
    }

    for (iTemp=0 ; iTemp<emissivityDb->nTemp ; iTemp++){
      demis = abservfunc(emissivityDb->dbTemp[iTemp], cellAlpha, freqData, emissivityDb->dbEmiss[iSpec][iTemp]);
      //printf("demis %d: %20.20lg\n",iTemp,demis );
      emissivityDb->dbEnerTemp[iSpec][iTemp] = demis;
      emissivityDb->dbLogEnerTemp[iSpec][iTemp] = log(demis);
    }
    free(cellAlpha);
  }
}

void setUpDbCumul(EmissivityDatabase* emissivityDb, FrequenciesData* freqData, int numSpec){
  int iSpec,iTemp,inu;
  double* dbCumul = (double*)malloc(sizeof(double)*(freqData->numFrequencies+1));
  double* diffEmis = (double*)malloc(sizeof(double)*freqData->numFrequencies);

  for (iSpec=0 ; iSpec<numSpec ; iSpec++){
    //for the first temperature
    iTemp = 0;
    dbCumul[0] = 0;
    for (inu=0 ; inu<freqData->numFrequencies ; inu++){
      dbCumul[inu+1]=dbCumul[inu] + emissivityDb->dbEmiss[iSpec][iTemp][inu] * freqData->frequenciesDnu[inu];
    }
    for (inu=0 ; inu<(freqData->numFrequencies+1) ; inu++){
      emissivityDb->dbCumulNorm[iSpec][iTemp][inu] = dbCumul[inu] / dbCumul[freqData->numFrequencies+1];
    }
    //for the rest temperatures
    for (iTemp=0 ; iTemp <emissivityDb->nTemp-1 ; iTemp++){
      for (inu=0 ; inu<freqData->numFrequencies ; inu++){
        diffEmis[inu] = emissivityDb->dbEmiss[iSpec][iTemp+1][inu] - emissivityDb->dbEmiss[iSpec][iTemp][inu];
      }
      dbCumul[0] = 0;
      for (inu=0 ; inu<freqData->numFrequencies ; inu++){
        dbCumul[inu+1] = dbCumul[inu] + diffEmis[inu]* freqData->frequenciesDnu[inu];
      }

      for (inu=0 ; inu<freqData->numFrequencies+1 ; inu++){
        emissivityDb->dbCumulNorm[iSpec][iTemp][inu] = dbCumul[inu] / dbCumul[freqData->numFrequencies];
      }
    }
  }
}

__global__ void convertEnergyToTemperature(DustTemperature* d_dustTemperature, DustDensity* d_dustDensity, Grid* d_grid, EmissivityDatabase* d_emissivityDb){
  unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
  //printf("totalPositions=%d\n",d_grid->totalCells);
  if (tid < d_grid->totalCells){
    int nxy = d_grid->nCoord[0]*d_grid->nCoord[1];
    int iz = tid/nxy;
    int idx1 = tid - iz*nxy;
    int iy = idx1/d_grid->nCoord[0];
    int ix = idx1-iy*d_grid->nCoord[0];
    //printf("tid=%d, ix,iy,iz=%d,%d,%d\n",tid,ix,iy,iz);
    double ener;
    for (int i=0 ; i<d_dustDensity->numSpec ; i++){
      ener = d_dustTemperature->cumulEner[i][iz][iy][ix]/(d_dustDensity->densities[i][iz][iy][ix]*d_grid->cellVolumes);
      //printf("ener=%10.10lg\n",ener);
      if (ener>0.0){
        d_dustTemperature->temperatures[i][iz][iy][ix] = computeDusttempEnergyBd(d_emissivityDb, ener, i);
        //printf("temp[%d,%d,%d] = %10.10lg\n",ix,iy,iz,d_dustTemperature->temperatures[i][iz][iy][ix]);
      }else{
        d_dustTemperature->temperatures[i][iz][iy][ix] = 0.0;
      }
    }
  }else{
    return;
  }
}

EmissivityDatabase* setUpEmissivityDatabase(float nTemp, float temp0, float temp1, DustOpacity* dustOpacity, FrequenciesData* freqData){
  printf("Set up emissivity database...\n");
  EmissivityDatabase* emissivityDb = allocateMemoryToEmissivityDatabase(nTemp, temp0, temp1, dustOpacity->numSpec, freqData->numFrequencies);
  setUpDbTemp(emissivityDb);
  setUpDbEnerTemp(emissivityDb,dustOpacity,freqData);
  setUpDbCumul(emissivityDb, freqData, dustOpacity->numSpec);
  return emissivityDb;
}
