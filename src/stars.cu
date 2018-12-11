#include "constants.cuh"
#include "global_functions.cuh"
#include "stars.cuh"

//CONSTANTS


Stars* allocateMemoryToStars(int numStars, int numLambdas){
  Stars* stars = (Stars*)malloc (sizeof(Stars));
  stars->numStars = numStars;
  stars->radius = (double*)malloc(sizeof(double)*numStars);
  stars->masses = (double*)malloc(sizeof(double)*numStars);
  stars->positions = (double**)malloc(sizeof(double*)*numStars);
  stars->spec = (double**)malloc(sizeof(double*)*numStars);
  stars->specCum = (double**)malloc(sizeof(double*)*numStars);
  stars->luminosities = (double*)malloc(sizeof(double)*numStars);
  stars->luminositiesCum = (double*)malloc(sizeof(double)*numStars);
  stars->energies = (double*)malloc(sizeof(double)*numStars);

  int i;
  for (i=0 ; i<numStars ; i++){
    stars->positions[i] = (double*)malloc(sizeof(double)*3);
    stars->spec[i] = (double*)malloc(sizeof(double)*numLambdas);
    stars->specCum[i] = (double*)malloc(sizeof(double)* (numLambdas+1));

  }

  return stars;
}

void deallocateStars(Stars* stars){
  free(stars->radius);
  free(stars->masses);
  free(stars->luminosities);
  free(stars->luminositiesCum);
  free(stars->energies);
  for (int i=0 ; i<stars->numStars ; i++){
    free(stars->positions[i]);
    free(stars->spec[i]);
    free(stars->specCum[i]);
  }
  free(stars->positions);
  free(stars->spec);
  free(stars->specCum);
  free(stars);
}

Stars* starsTransferToDevice(Stars* h_stars, int numLambdas){
  printf("Transfer stars to device...\n");
  double* radius;
  double* masses;
  double** positions;
  double** spec;
  double** specCum;
  double* luminosities;
  double* luminositiesCum;
  double* energies;
  int numStars = h_stars->numStars;

  double** supportSpec;
  double** supportCum;
  double** supportPositions;

  Stars* d_stars;
  cudaMalloc( (void**)&(d_stars), sizeof(Stars) );
  cudaMalloc((void**)&radius, sizeof(double)*numStars);
  cudaMalloc((void**)&masses, sizeof(double)*numStars);
  cudaMalloc((void**)&luminosities, sizeof(double)*numStars);
  cudaMalloc((void**)&luminositiesCum, sizeof(double)*numStars);
  cudaMalloc((void**)&energies, sizeof(double)*numStars);

  cudaMemcpy(d_stars, h_stars, sizeof(Stars), cudaMemcpyHostToDevice);
  cudaMemcpy(&(d_stars->radius), &radius, sizeof(double *), cudaMemcpyHostToDevice);
  cudaMemcpy(&(d_stars->masses), &masses, sizeof(double *), cudaMemcpyHostToDevice);
  cudaMemcpy(&(d_stars->luminosities), &luminosities, sizeof(double *), cudaMemcpyHostToDevice);
  cudaMemcpy(&(d_stars->luminositiesCum), &luminositiesCum, sizeof(double *), cudaMemcpyHostToDevice);
  cudaMemcpy(&(d_stars->energies), &energies, sizeof(double *), cudaMemcpyHostToDevice);

  cudaMemcpy(radius, h_stars->radius, sizeof(double)*numStars, cudaMemcpyHostToDevice);
  cudaMemcpy(masses, h_stars->masses, sizeof(double)*numStars, cudaMemcpyHostToDevice);
  cudaMemcpy(luminosities, h_stars->luminosities, sizeof(double)*numStars, cudaMemcpyHostToDevice);
  cudaMemcpy(luminositiesCum, h_stars->luminositiesCum, sizeof(double)*numStars, cudaMemcpyHostToDevice);
  cudaMemcpy(energies, h_stars->energies, sizeof(double)*numStars, cudaMemcpyHostToDevice);

  //positions, spec, specCum
  cudaMalloc((void **) &positions,numStars*sizeof(double*));
  cudaMalloc((void **) &spec,numStars*sizeof(double*));
  cudaMalloc((void **) &specCum,numStars*sizeof(double*));
  supportSpec =(double**) malloc(sizeof(double*)*numStars);
  supportCum =(double**) malloc(sizeof(double*)*numStars);
  supportPositions =(double**) malloc(sizeof(double*)*numStars);

  for (int i=0;i<numStars;i++){
    cudaMalloc((void**) &supportPositions[i],sizeof(double)*3);
    cudaMalloc((void**) &supportSpec[i],sizeof(double)*numLambdas);
    cudaMalloc((void**) &supportCum[i],sizeof(double)* (numLambdas+1));
    cudaMemcpy(supportPositions[i],h_stars->positions[i],3*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(supportSpec[i],h_stars->spec[i],numLambdas*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(supportCum[i],h_stars->specCum[i],(numLambdas+1)*sizeof(double),cudaMemcpyHostToDevice);
  }
  cudaMemcpy(positions,supportPositions,sizeof(double*)*numStars,cudaMemcpyHostToDevice);
  cudaMemcpy(spec,supportSpec,sizeof(double*)*numStars,cudaMemcpyHostToDevice);
  cudaMemcpy(specCum,supportCum,sizeof(double*)*numStars,cudaMemcpyHostToDevice);

  cudaMemcpy(&(d_stars->positions), &positions, sizeof(double**), cudaMemcpyHostToDevice);
  cudaMemcpy(&(d_stars->spec), &spec, sizeof(double**), cudaMemcpyHostToDevice);
  cudaMemcpy(&(d_stars->specCum), &specCum, sizeof(double**), cudaMemcpyHostToDevice);

  free(supportSpec);
  free(supportCum);
  free(supportPositions);

  return d_stars;
}

Stars* readStars(FrequenciesData* freqData){
  double lambdaTmp;
  Stars* stars;
  char line[30];
  int numStars;
  int numLambdas;
  const char* nameFile = "inputs/stars.inp";
  FILE* starsFile = fopen(nameFile, "r");
  if (starsFile == NULL){
    printf("Failed to open stars.inp. Maybe don't exist\n");
    exit(1);
  }else{
    fscanf(starsFile, "%d %d\n", &numStars,&numLambdas);

    if (numStars < 0){
      printf("ERROR: nr star must be > 0");
      exit(1);
    }

    if (freqData->numFrequencies != numLambdas){
      printf("ERROR: nr of frequency points in stars.inp is unequal\n");
      printf("to that in wavelength_micron.inp\n");
      exit(1);
    }
    stars = allocateMemoryToStars(numStars, numLambdas);

    int i,j;
    for (i=0 ; i<numStars;i++){
      fscanf(starsFile, "%lf %lf %lf %lf %lf\n",
      &stars->radius[i], &stars->masses[i], &stars->positions[i][0], &stars->positions[i][1], &stars->positions[i][2]);
      //printf("radius=%10.10lg, masses=%10.10lg, positions=%10.10lg %10.10lg %10.10lg \n",
      //stars->radius[i], stars->masses[i], stars->positions[i][0], stars->positions[i][1], stars->positions[i][2]);
    }

    for (i=0 ; i<numLambdas ; i++){
      fgets(line, 30, starsFile);
      lambdaTmp = atof(line);
      lambdaTmp = 10000 * C / lambdaTmp;
      //COMPROBAR QUUE LAMBDA == FREQUENCY
    }

    //now read all the spectra
    double dummy;
    for (i=0 ; i<numStars ; i++){
      printf("Star %d\n",i);
      fgets(line, 30, starsFile);
      dummy = atof(line);
      printf("dummy=%lf\n",dummy);

      // IMPORTANT: If the first number is negative, then instead of reading
      //            the entire spectrum, a blackbody is taken with a temperature
      //           equal to the absolute value of that first number.
      if (dummy < 0){
        //make blackbody star
        for (j=0 ; j<numLambdas ; j++){
          stars->spec[i][j]=blackbodyPlanck(fabs(dummy),freqData->frequencies[j]);
        }
      }else{

        // Read the full intensity spectrum. Note that the spectrum
        // is given in the usual flux-at-one-parsec format, so we have
        // to translate this here to intensity-at-stellar-surface.
        stars->spec[i][0] = 3.0308410e36 * dummy / (stars->radius[i]*stars->radius[i]);
        for (j=1 ; j<numLambdas;j++){
          fgets(line, 30, starsFile);
          dummy = atof(line);
          stars->spec[i][j] = 3.0308410e36 * dummy / (stars->radius[i]*stars->radius[i]);
        }
      }
    }
    fclose(starsFile);
  }
  return stars;
}

void calculateCumulativeSpec(Stars* stars, FrequenciesData* freqData){
  int iStar,j;
  for (iStar=0 ; iStar<stars->numStars ; iStar++){
    stars->specCum[iStar][0] = 0;
    for (j=0 ; j<freqData->numFrequencies ; j++){
      stars->specCum[iStar][j+1] = stars->specCum[iStar][j] + stars->spec[iStar][j] * freqData->frequenciesDnu[j];
    }

    for (j=0 ; j<freqData->numFrequencies+1 ; j++){
      stars->specCum[iStar][j] = stars->specCum[iStar][j] / stars->specCum[iStar][freqData->numFrequencies];
    }
  }
}

void calculateStarsLuminosities(Stars* stars, FrequenciesData* freqData){
  int iStar,iFreq;
  double dum;
  stars->luminositiesTotal = 0;
  for (iStar=0 ; iStar<stars->numStars ; iStar++){
    stars->luminosities[iStar] = 0;
    for (iFreq=0 ; iFreq<freqData->numFrequencies ; iFreq++){
      dum = 4*PI*PI*stars->spec[iStar][iFreq] * stars->radius[iStar]*stars->radius[iStar];
      stars->luminosities[iStar] += dum * freqData->frequenciesDnu[iFreq];
    }

    stars->luminositiesTotal += stars->luminosities[iStar];
  }

  if (stars->luminositiesTotal > 0){
    stars->luminositiesCum[0] = 0;
    for (iStar=0 ; iStar<stars->numStars ; iStar++){
      stars->luminositiesCum[iStar+1] = stars->luminosities[iStar]/stars->luminositiesTotal;
    }
    stars->luminositiesCum[stars->numStars] = 1;
  }
}

void calculateEnergyStars(Stars* stars, int numPhotons){
  //numero fuentes de energia, en este caso, solo estrellas
  int numStarSrc = 0;
  int iStar;
  for (iStar=0 ; iStar<stars->numStars; iStar++){
    if (stars->luminosities[iStar] > 0){
      numStarSrc++;
    }
  }

  for (iStar=0 ; iStar<stars->numStars; iStar++){
    stars->energies[iStar] = stars->luminosities[iStar] * numStarSrc / numPhotons;
  }
}

void jitterStars(Stars* stars, Grid* grid){
  double smallNumberX = 0.48957949203816064943E-12;
  double smallNumberY = 0.38160649492048957394E-12;
  double smallNumberZ = 0.64943484920957938160E-12;

  double szx = grid->cellWallsX[grid->nCoord[0]]-grid->cellWallsX[0];//nx
  double szy = grid->cellWallsY[grid->nCoord[1]]-grid->cellWallsY[0];//ny
  double szz = grid->cellWallsZ[grid->nCoord[2]]-grid->cellWallsZ[0];//nz

  int iStar;
  for (iStar=0 ; iStar<stars->numStars ; iStar++){
    stars->positions[iStar][0] = stars->positions[iStar][0] + szx*smallNumberX;
    stars->positions[iStar][1] = stars->positions[iStar][1] + szy*smallNumberY;
    stars->positions[iStar][2] = stars->positions[iStar][2] + szz*smallNumberZ;
  }
}

Stars* setUpStars(FrequenciesData* freqData, Grid* grid, int numPhotons){
  printf("Set up stars...\n");
  Stars* stars = readStars(freqData);
  calculateCumulativeSpec(stars,freqData);
  calculateStarsLuminosities(stars, freqData);
  calculateEnergyStars(stars, numPhotons);
  jitterStars(stars, grid);
  return stars;
}
