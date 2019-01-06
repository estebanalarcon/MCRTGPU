#include <string.h>
#include "opacities.cuh"
#include "global_functions.cuh"

DustOpacity* allocateMemoryToDustOpacity(int iFormat, int numSpec){
  DustOpacity* dustOpacity = (DustOpacity*)malloc(sizeof(DustOpacity));
  dustOpacity->iFormat = iFormat;
  dustOpacity->numSpec = numSpec;

  dustOpacity->inputStyle = (int*)malloc(sizeof(int)*numSpec);
  dustOpacity->iQuantums = (int*)malloc(sizeof(int)*numSpec);
  dustOpacity->numLambdas = (int*)malloc(sizeof(int)*numSpec);

  dustOpacity->lambdas = (double**)malloc(sizeof(double*)*numSpec);
  dustOpacity->kappaA = (double**)malloc(sizeof(double*)*numSpec);
  dustOpacity->kappaS = (double**)malloc(sizeof(double*)*numSpec);
  dustOpacity->g = (double**)malloc(sizeof(double*)*numSpec);
  return dustOpacity;
}

void allocateMemoryToKappas(DustOpacity* dustOpacity, int numLambdas, int iSpec){
  dustOpacity->numLambdas[iSpec] = numLambdas;
  dustOpacity->lambdas[iSpec] = (double*)malloc(sizeof(double)*numLambdas);
  dustOpacity->kappaA[iSpec] = (double*)malloc(sizeof(double)*numLambdas);
  dustOpacity->kappaS[iSpec] = (double*)malloc(sizeof(double)*numLambdas);
  dustOpacity->g[iSpec] = (double*)malloc(sizeof(double)*numLambdas);
}

void deallocateDustOpacities(DustOpacity* dustOpacity){
  for (int i=0 ; i<dustOpacity->numSpec ; i++){
    free(dustOpacity->lambdas[i]);
    free(dustOpacity->kappaA[i]);
    free(dustOpacity->kappaS[i]);
    free(dustOpacity->g[i]);
  }

  free(dustOpacity->inputStyle);
  free(dustOpacity->iQuantums);
  free(dustOpacity->numLambdas);
  free(dustOpacity);
}

DustOpacity* dustOpacityTransferToDevice(DustOpacity* h_dustOpacity, int numFreq){
  printf("Transfer DustOpacity to device...\n");
  DustOpacity* d_dustOpacity;
  int numSpec = h_dustOpacity->numSpec;
  int* inputStyle;
  int* iQuantums;
  int* numLambdas;

  double** lambdas;
  double** kappaA;
  double** kappaS;
  double** g;
  double** supportLam;
  double** supportKA;
  double** supportKS;
  double** supportG;

  cudaMalloc( (void**)&(d_dustOpacity), sizeof(DustOpacity) );
  cudaMalloc((void**)&inputStyle, sizeof(int)*numSpec);
  cudaMalloc((void**)&iQuantums, sizeof(int)*numSpec);
  cudaMalloc((void**)&numLambdas, sizeof(int)*numSpec);

  cudaMemcpy(d_dustOpacity, h_dustOpacity, sizeof(DustOpacity), cudaMemcpyHostToDevice);
  cudaMemcpy(&(d_dustOpacity->inputStyle), &inputStyle, sizeof(int *), cudaMemcpyHostToDevice);
  cudaMemcpy(&(d_dustOpacity->iQuantums), &iQuantums, sizeof(int *), cudaMemcpyHostToDevice);
  cudaMemcpy(&(d_dustOpacity->numLambdas), &numLambdas, sizeof(int *), cudaMemcpyHostToDevice);

  cudaMemcpy(inputStyle, h_dustOpacity->inputStyle, sizeof(int)*numSpec, cudaMemcpyHostToDevice);
  cudaMemcpy(iQuantums, h_dustOpacity->iQuantums, sizeof(int)*numSpec, cudaMemcpyHostToDevice);
  cudaMemcpy(numLambdas, h_dustOpacity->numLambdas, sizeof(int)*numSpec, cudaMemcpyHostToDevice);

  cudaMalloc((void **) &lambdas,numSpec*sizeof(double*));
  cudaMalloc((void **) &kappaA,numSpec*sizeof(double*));
  cudaMalloc((void **) &kappaS,numSpec*sizeof(double*));
  cudaMalloc((void **) &g,numSpec*sizeof(double*));

  supportLam =(double**) malloc(sizeof(double*)*numSpec);
  supportKA =(double**) malloc(sizeof(double*)*numSpec);
  supportKS =(double**) malloc(sizeof(double*)*numSpec);
  supportG =(double**) malloc(sizeof(double*)*numSpec);


  for (int i=0;i<numSpec;i++){
    cudaMalloc((void**) &supportLam[i],sizeof(double)*numFreq);
    cudaMalloc((void**) &supportKA[i],sizeof(double)*numFreq);
    cudaMalloc((void**) &supportKS[i],sizeof(double)*numFreq);
    cudaMalloc((void**) &supportG[i],sizeof(double)*numFreq);

    cudaMemcpy(supportLam[i],h_dustOpacity->lambdas[i],numFreq*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(supportKA[i],h_dustOpacity->kappaA[i],numFreq*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(supportKS[i],h_dustOpacity->kappaS[i],numFreq*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(supportG[i],h_dustOpacity->g[i],numFreq*sizeof(double),cudaMemcpyHostToDevice);

  }
  cudaMemcpy(lambdas,supportLam,sizeof(double*)*numSpec,cudaMemcpyHostToDevice);
  cudaMemcpy(kappaA,supportKA,sizeof(double*)*numSpec,cudaMemcpyHostToDevice);
  cudaMemcpy(kappaS,supportKS,sizeof(double*)*numSpec,cudaMemcpyHostToDevice);
  cudaMemcpy(g,supportG,sizeof(double*)*numSpec,cudaMemcpyHostToDevice);

  cudaMemcpy(&(d_dustOpacity->lambdas), &lambdas, sizeof(double**), cudaMemcpyHostToDevice);
  cudaMemcpy(&(d_dustOpacity->kappaA), &kappaA, sizeof(double**), cudaMemcpyHostToDevice);
  cudaMemcpy(&(d_dustOpacity->kappaS), &kappaS, sizeof(double**), cudaMemcpyHostToDevice);
  cudaMemcpy(&(d_dustOpacity->g), &g, sizeof(double**), cudaMemcpyHostToDevice);

  free(supportLam);
  free(supportKA);
  free(supportKS);
  free(supportG);

  return d_dustOpacity;
}

void checkOpacities(DustOpacity* dustOpacity, int iSpec, char* nameFile, int numLambdas){
  int sgn, i;
  double dum;
  // Check that the lambda array is monotonically increasing/decreasing
  if (dustOpacity->lambdas[iSpec][1] > dustOpacity->lambdas[iSpec][0]){
    sgn=1;
  }else if (dustOpacity->lambdas[iSpec][1] < dustOpacity->lambdas[iSpec][0]){
    sgn=-1;
  }else{
    printf("ERROR: while reading opacity file %s. Cannot have twice the same lambda.\n", nameFile);
    exit(1);
  }
  for (i=2 ; i<numLambdas; i++){
    dum = (dustOpacity->lambdas[iSpec][i]-dustOpacity->lambdas[iSpec][i-1]) * sgn;
    if (dum == 0){
      printf("ERROR: while reading opacity file %s. Cannot have twice the same lambda.\n", nameFile);
      exit(1);
    }else if (dum < 0){
      printf("ERROR: while reading opacity file %s. Wavelength array is not monotonic.\n", nameFile);
      exit(1);
    }
  }

  //Check that the opacities are non-zero
  for (i=0 ; i<numLambdas ; i++){
    if (dustOpacity->kappaA[iSpec][i] < 0){
      printf("ERROR: while reading opacity file %s. Negative kappa_a detected at ilam=%d\n", nameFile, i);
      exit(1);
    }

    if (dustOpacity->kappaS[iSpec][i] < 0){
      printf("ERROR: while reading opacity file %s. Negative kappa_s detected at ilam=%d\n", nameFile, i);
      exit(1);
    }
  }
  return;
}

void convertLambdasToFrequency(DustOpacity* dustOpacity){

  int iSpec,iLam;
  for (iSpec=0 ; iSpec < dustOpacity->numSpec ; iSpec++){
    for (iLam=0 ; iLam<dustOpacity->numLambdas[iSpec]; iLam++){
      dustOpacity->lambdas[iSpec][iLam] = 2.9979e14 / dustOpacity->lambdas[iSpec][iLam];
    }
  }
}

void remapOpacitiesValues(DustOpacity* dustOpacity, FrequenciesData* freqData){
  int i;
  double *remapResult;
  for (i=0 ; i<dustOpacity->numSpec ; i++){
    //kappaAbs
    //printf("KAPPA ABS=%10.10lf\n",dustOpacity->kappaA[i][0]);
    //printf("numLambdas=%d\n",dustOpacity->numLambdas[i]);
    remapResult = remapFunction(dustOpacity->numLambdas[i], dustOpacity->lambdas[i],
                  dustOpacity->kappaA[i], freqData->numFrequencies, freqData->frequencies,2);
    free(dustOpacity->kappaA[i]);
    dustOpacity->kappaA[i] = (double*)malloc(sizeof(double)*freqData->numFrequencies);
    //dustOpacity->kappaA[i] = remapResult;
    memcpy(dustOpacity->kappaA[i], remapResult, sizeof(double)*freqData->numFrequencies);
    free(remapResult);
    //kappaScat
    remapResult = remapFunction(dustOpacity->numLambdas[i], dustOpacity->lambdas[i],
                  dustOpacity->kappaS[i], freqData->numFrequencies, freqData->frequencies,2);
    free(dustOpacity->kappaS[i]);
    dustOpacity->kappaS[i] = remapResult;

    //G anisotropic
    remapResult = remapFunction(dustOpacity->numLambdas[i], dustOpacity->lambdas[i],
                  dustOpacity->g[i], freqData->numFrequencies, freqData->frequencies,1);
    free(dustOpacity->g[i]);
    dustOpacity->g[i] = remapResult;
  }
}

void readDustKappa(char* nameDust, DustOpacity* dustOpacity, int iSpec){
  int iFormat, numLambdas;
  char nameFile[100];
  char line[300];
  char s[2] = " ";
  char* token;
  const char* startFile = "inputs/dustkappa_";
  const char* endFile = ".inp";
  strcpy(nameFile, startFile);
  strcat(nameFile,nameDust);
  strcat(nameFile,endFile);
  FILE* kappaFile = fopen(nameFile, "r");
  if (kappaFile == NULL){
    printf("Failed to open %s. Maybe don't exist\n", nameFile);
    exit(1);
  }else{
    printf("Reading %s\n",nameFile);
    fgets(line, 30, kappaFile);
    iFormat = atoi(line);
    fgets(line, 30, kappaFile);
    numLambdas = atoi(line);
    allocateMemoryToKappas(dustOpacity, numLambdas, iSpec);
    int i;
    printf("iFormat %d\n",iFormat);
    //read lambda and kappaAbs
    if (iFormat==1){
      for (i=0 ; i<numLambdas ; i++){
        fscanf(kappaFile,"%[^\n]\n", line);
        //printf("%s\n",line);
        token = strtok(line, s);
        dustOpacity->lambdas[iSpec][i]=atof(token);
        token = strtok(NULL, s);
        dustOpacity->kappaA[iSpec][i]=atof(token);
        dustOpacity->kappaS[iSpec][i]=0.0;
        dustOpacity->g[iSpec][i]=0.0;
        //printf("lambdas=%g kappaA=%g kappaS=%g g=%g\n",
        //dustOpacity->lambdas[iSpec][i], dustOpacity->kappaA[iSpec][i], dustOpacity->kappaS[iSpec][i],dustOpacity->g[iSpec][i]);
      }
    }//read lambda,kappaAbs,kappaScat
    else if (iFormat==2){
      for (i=0 ; i<numLambdas ; i++){
        fscanf(kappaFile,"%[^\n]\n", line);
        token = strtok(line, s);
        dustOpacity->lambdas[iSpec][i]=atof(token);
        token = strtok(NULL, s);
        dustOpacity->kappaA[iSpec][i]=atof(token);
        token = strtok(NULL, s);
        dustOpacity->kappaS[iSpec][i]=atof(token);
        dustOpacity->g[iSpec][i]=0.0;
        //printf("lambdas=%g kappaA=%g kappaS=%g g=%g\n",
        //dustOpacity->lambdas[iSpec][i], dustOpacity->kappaA[iSpec][i], dustOpacity->kappaS[iSpec][i],dustOpacity->g[iSpec][i]);

      }
    }//read lambda,kappaAbs,kappaScat, g
    else if (iFormat ==3){
      for (i=0 ; i<numLambdas ; i++){
        fscanf(kappaFile,"%[^\n]\n", line);
        token = strtok(line, s);
        dustOpacity->lambdas[iSpec][i]=atof(token);
        token = strtok(NULL, s);
        dustOpacity->kappaA[iSpec][i]=atof(token);
        token = strtok(NULL, s);
        dustOpacity->kappaS[iSpec][i]=atof(token);
        token = strtok(NULL, s);
        dustOpacity->g[iSpec][i]=atof(token);
        //printf("lambdas=%g kappaA=%g kappaS=%g g=%g\n",
        //dustOpacity->lambdas[iSpec][i], dustOpacity->kappaA[iSpec][i], dustOpacity->kappaS[iSpec][i],dustOpacity->g[iSpec][i]);
      }
    }else{
      printf("ERROR: iformat in %s not support\n",nameFile);
      exit(1);
    }
    fclose(kappaFile);
    checkOpacities(dustOpacity, iSpec, nameFile, numLambdas);
  }
}

DustOpacity* readDustOpacity(){
  DustOpacity* dustOpacity;
  int iFormat, numSpec;
  char line[100];
  const char* nameFile = "inputs/dustopac.inp";
  FILE* opacFile = fopen(nameFile, "r");
  if (opacFile == NULL){
    printf("Failed to open dustopac.inp. Maybe don't exist\n");
    exit(1);
  }else{

    fgets(line, 100, opacFile);
    iFormat = atoi(line);
    fgets(line, 100, opacFile);
    numSpec = atoi(line);
    printf("Number species:%d\n",numSpec);

    //VERIFICAR NUMSPEC SEA IGUAL AL DE DUST_DENSITY. FALTA

    dustOpacity = allocateMemoryToDustOpacity(iFormat, numSpec);
    int iSpec;
    for (iSpec=0 ; iSpec<numSpec ; iSpec++){
      fgets(line, 100, opacFile);//ELIMINAR LINEA SEPARADORA
      fgets(line, 100, opacFile);
      dustOpacity->inputStyle[iSpec]=atoi(line);
      fgets(line, 100, opacFile);
      dustOpacity->iQuantums[iSpec]=atoi(line);
      fgets(line, 100, opacFile);//NAME OF DUST
      strtok(line, "\n");
      readDustKappa(line, dustOpacity, iSpec);
    }
  }
  return dustOpacity;
}

DustOpacity* setUpDustOpacity(FrequenciesData* freqData){
  printf("Set up opacities...\n");
  DustOpacity* dustOpacity = readDustOpacity();
  convertLambdasToFrequency(dustOpacity);
  remapOpacitiesValues(dustOpacity, freqData);
  return dustOpacity;
}
