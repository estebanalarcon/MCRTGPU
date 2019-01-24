#include "dust.cuh"

__host__ DustTemperature* allocateMemoryToDustTemperature(int numSpec, Grid* grid){
  int nx = grid->nCoord[0];
  int ny = grid->nCoord[1];
  int nz = grid->nCoord[2];
  DustTemperature* dustTemperature = (DustTemperature*)malloc(sizeof(DustTemperature));
  dustTemperature->cumulEner = (double****)malloc(sizeof(double***)*numSpec);
  dustTemperature->temperatures = (double****)malloc(sizeof(double***)*numSpec);
  int j,k,iSpec;
  for (iSpec=0 ; iSpec<numSpec ; iSpec++){
    dustTemperature->cumulEner[iSpec] = (double***)malloc(sizeof(double**)*nz);
    dustTemperature->temperatures[iSpec] = (double***)malloc(sizeof(double**)*nz);
    for (k=0 ; k<nz ; k++){
      dustTemperature->cumulEner[iSpec][k] = (double**)malloc(sizeof(double*)*ny);
      dustTemperature->temperatures[iSpec][k] = (double**)malloc(sizeof(double*)*ny);
      for (j=0 ; j<ny ; j++){
        dustTemperature->cumulEner[iSpec][k][j] = (double*)malloc(sizeof(double)*nx);
        dustTemperature->temperatures[iSpec][k][j] = (double*)malloc(sizeof(double)*nx);
      }
    }
  }
  return dustTemperature;
}

__host__ void deallocateCumulEner(DustTemperature* dustTemperature, int numSpec, int ny, int nz){
  for (int iSpec=0 ; iSpec<numSpec ; iSpec++){
    for (int k=0 ; k<nz ; k++){
      for (int j=0 ; j<ny ; j++){
        free(dustTemperature->cumulEner[iSpec][k][j]);
      }
      free(dustTemperature->cumulEner[iSpec][k]);
    }
    free(dustTemperature->cumulEner[iSpec]);
  }
  free(dustTemperature->cumulEner);
}

__host__ void deallocateDustTemperature(DustTemperature* dustTemperature, int numSpec, int ny, int nz){
  printf("Deallocate dust temperature...\n");
  for (int iSpec=0 ; iSpec<numSpec ; iSpec++){
    for (int k=0 ; k<nz ; k++){
      for (int j=0 ; j<ny ; j++){
        free(dustTemperature->temperatures[iSpec][k][j]);
      }
      printf("free dustTemperature->temperatures[iSpec][k])\n");
      free(dustTemperature->temperatures[iSpec][k]);
    }
    free(dustTemperature->temperatures[iSpec]);
  }
  printf("free dustTemperature->temperatures\n");
  free(dustTemperature->temperatures);
  printf("free dustTemperature\n");
  free(dustTemperature);
}

__host__ void inicializeDustTemperature(DustTemperature* dustTemperature, DustDensity* dustDensity){
  dustTemperature->totalPositions = dustDensity->nx*dustDensity->ny*dustDensity->nz;
  for (int iSpec=0 ; iSpec<dustDensity->numSpec ; iSpec++){
    for (int k=0 ; k<dustDensity->nz ; k++){
      for (int j=0 ; j<dustDensity->ny ; j++){
        for (int i=0 ; i<dustDensity->nx ; i++){
          dustTemperature->cumulEner[iSpec][k][j][i] = 0;
          dustTemperature->temperatures[iSpec][k][j][i] = 0;
        }
      }
    }
  }
  return;
}

__host__ DustDensity* allocateMemoryToDustDensity(int iFormat, int numCells, int numSpec, Grid* grid){
  int nx = grid->nCoord[0];
  int ny = grid->nCoord[1];
  int nz = grid->nCoord[2];
  DustDensity* dustDensity = (DustDensity*)malloc(sizeof(DustDensity));

  dustDensity->iFormat = iFormat;
  dustDensity->numCells = numCells;
  dustDensity->numSpec = numSpec;
  dustDensity->nx = nx;
  dustDensity->ny = ny;
  dustDensity->nz = nz;

  dustDensity->densities = (double****)malloc(sizeof(double***)*numSpec);
  int j,k,iSpec;
  for (iSpec=0 ; iSpec<numSpec ; iSpec++){
    dustDensity->densities[iSpec] = (double***)malloc(sizeof(double**)*nz);
    for (k=0 ; k<nz ; k++){
      dustDensity->densities[iSpec][k] = (double**)malloc(sizeof(double*)*ny);
      for (j=0 ; j<ny ; j++){
        dustDensity->densities[iSpec][k][j] = (double*)malloc(sizeof(double)*nx);
      }
    }
  }
  return dustDensity;
}

__host__ void deallocateDensities(DustDensity* dustDensity){
  for (int iSpec=0 ; iSpec<dustDensity->numSpec ; iSpec++){
    for (int z=0 ; z<dustDensity->nz ; z++){
      for (int y=0 ; y<dustDensity->ny ; y++){
        free(dustDensity->densities[iSpec][z][y]);
      }
      free(dustDensity->densities[iSpec][z]);
    }
    free(dustDensity->densities[iSpec]);
  }
  free(dustDensity->densities);
  free(dustDensity);
}

__host__ DustDensity* dustDensityTransferToDevice(DustDensity* h_dustDensity){
  printf("Transfer DustDensity to device...\n");
  DustDensity* d_dustDensity;
  int numSpec = h_dustDensity->numSpec;
  int nz = h_dustDensity->nz;
  int ny = h_dustDensity->ny;
  int nx = h_dustDensity->nx;

  double**** densities;
  double**** support4;
  double*** support3;
  double** support2;

  cudaMalloc((void **) &densities,numSpec*sizeof(double***));
  support4 =(double****) malloc(sizeof(double***)*numSpec);
  for (int iSpec=0 ; iSpec<numSpec ; iSpec++){
    cudaMalloc((void**) &support4[iSpec],sizeof(double**)*nz);
    support3 =(double***) malloc(sizeof(double**)*nz);
    for (int k=0;k<nz;k++){
      cudaMalloc((void**) &support3[k],sizeof(double*)*ny);
      support2 =(double**) malloc(sizeof(double*)*ny);
      for (int j=0;j<ny;j++){
        cudaMalloc((void**) &support2[j],sizeof(double)*nx);
        cudaMemcpy(support2[j],h_dustDensity->densities[iSpec][k][j],nx*sizeof(double),cudaMemcpyHostToDevice);
      }
      cudaMemcpy(support3[k],support2,sizeof(double*)*ny,cudaMemcpyHostToDevice);
      free(support2);
    }
    cudaMemcpy(support4[iSpec],support3,sizeof(double**)*nz,cudaMemcpyHostToDevice);
    free(support3);
  }
  cudaMemcpy(densities,support4,sizeof(double***)*numSpec,cudaMemcpyHostToDevice);
  free(support4);

  cudaMalloc((void **) &d_dustDensity,sizeof(DustDensity));
  cudaMemcpy(d_dustDensity, h_dustDensity, sizeof(DustDensity), cudaMemcpyHostToDevice);
  cudaMemcpy(&(d_dustDensity->densities), &densities, sizeof(double****), cudaMemcpyHostToDevice);

  return d_dustDensity;
}

__host__ void writeDustTemperature(DustTemperature* h_dustTemperature, DustTemperature* d_dustTemperature, int numSpec, int nz, int ny, int nx){
  printf("Writing DustTemperature file...\n");
  int totalPositions = nz*ny*nx;
  FILE* fileTemperature = fopen("dust_temperature_gpu.dat","w");
  fprintf(fileTemperature,"%d\n",1);
  fprintf(fileTemperature,"%d\n",totalPositions);
  fprintf(fileTemperature,"%d\n",numSpec);

  double**** support4temp;
  double*** support3temp;
  double** support2temp;
  double* support1temp;
  double**** temperatures;
  temperatures = (double****) malloc(sizeof(double***)*numSpec);
  cudaMemcpy(h_dustTemperature,d_dustTemperature,sizeof(DustTemperature),cudaMemcpyDeviceToHost);
  support4temp =(double****) malloc(sizeof(double***)*numSpec);
  cudaMemcpy(support4temp,h_dustTemperature->temperatures,numSpec*sizeof(double***),cudaMemcpyDeviceToHost);
  for (int iSpec=0 ; iSpec<numSpec ; iSpec++){
    support3temp =(double***) malloc(sizeof(double**)*nz);
    cudaMemcpy(support3temp,support4temp[iSpec],nz*sizeof(double**),cudaMemcpyDeviceToHost);
    for (int k=0 ; k<nz ; k++){
      support2temp =(double**) malloc(sizeof(double*)*ny);
      cudaMemcpy(support2temp,support3temp[k],ny*sizeof(double*),cudaMemcpyDeviceToHost);
      for (int j=0 ; j<ny ; j++){
        support1temp = (double*) malloc(sizeof(double)*nx);
        cudaMemcpy(support1temp,support2temp[j],nx*sizeof(double),cudaMemcpyDeviceToHost);
        for (int i=0 ; i<nx ; i++){
          fprintf(fileTemperature,"%10.10lg\n",support1temp[i]);
        }
        free(support1temp);
      }
      free(support2temp);
    }
    free(support3temp);
  }
  free(support4temp);
  free(temperatures);
  fclose(fileTemperature);
}

DustTemperature* dustTemperatureTransferToDevice(DustTemperature* h_dustTemperature, DustDensity* h_dustDensity){
  printf("Transfer DustTemperature to device...\n");
  DustTemperature* d_dustTemperature;
  int numSpec = h_dustDensity->numSpec;
  int nz = h_dustDensity->nz;
  int ny = h_dustDensity->ny;
  int nx = h_dustDensity->nx;

  double**** temperatures;
  double**** cumulEner;

  double**** support4temp;
  double*** support3temp;
  double** support2temp;

  double**** support4Ener;
  double*** support3Ener;
  double** support2Ener;

  cudaMalloc((void **) &temperatures,numSpec*sizeof(double***));
  cudaMalloc((void **) &cumulEner,numSpec*sizeof(double***));

  support4temp =(double****) malloc(sizeof(double***)*numSpec);
  support4Ener =(double****) malloc(sizeof(double***)*numSpec);
  for (int iSpec=0 ; iSpec<numSpec ; iSpec++){
    cudaMalloc((void**) &support4temp[iSpec],sizeof(double**)*nz);
    cudaMalloc((void**) &support4Ener[iSpec],sizeof(double**)*nz);
    support3temp =(double***) malloc(sizeof(double**)*nz);
    support3Ener =(double***) malloc(sizeof(double**)*nz);
    for (int k=0;k<nz;k++){
      cudaMalloc((void**) &support3temp[k],sizeof(double*)*ny);
      cudaMalloc((void**) &support3Ener[k],sizeof(double*)*ny);
      support2temp =(double**) malloc(sizeof(double*)*ny);
      support2Ener =(double**) malloc(sizeof(double*)*ny);
      for (int j=0;j<ny;j++){
        cudaMalloc((void**) &support2temp[j],sizeof(double)*nx);
        cudaMalloc((void**) &support2Ener[j],sizeof(double)*nx);
        cudaMemcpy(support2temp[j],h_dustTemperature->temperatures[iSpec][k][j],nx*sizeof(double),cudaMemcpyHostToDevice);
        cudaMemcpy(support2Ener[j],h_dustTemperature->cumulEner[iSpec][k][j],nx*sizeof(double),cudaMemcpyHostToDevice);
      }
      cudaMemcpy(support3temp[k],support2temp,sizeof(double*)*ny,cudaMemcpyHostToDevice);
      cudaMemcpy(support3Ener[k],support2Ener,sizeof(double*)*ny,cudaMemcpyHostToDevice);
      free(support2temp);
      free(support2Ener);
    }
    cudaMemcpy(support4temp[iSpec],support3temp,sizeof(double**)*nz,cudaMemcpyHostToDevice);
    cudaMemcpy(support4Ener[iSpec],support3Ener,sizeof(double**)*nz,cudaMemcpyHostToDevice);
    free(support3temp);
    free(support3Ener);
  }
  cudaMemcpy(temperatures,support4temp,sizeof(double***)*numSpec,cudaMemcpyHostToDevice);
  cudaMemcpy(cumulEner,support4Ener,sizeof(double***)*numSpec,cudaMemcpyHostToDevice);
  free(support4temp);
  free(support4Ener);

  cudaMalloc((void **) &d_dustTemperature,sizeof(DustTemperature));
  cudaMemcpy(&(d_dustTemperature->temperatures), &temperatures, sizeof(double****), cudaMemcpyHostToDevice);
  cudaMemcpy(&(d_dustTemperature->cumulEner), &cumulEner, sizeof(double****), cudaMemcpyHostToDevice);

  return d_dustTemperature;
}

DustDensity* setUpDustDensity(Grid* grid){
  printf("Set up dustDensity...\n");
  DustDensity* dustDensity;
  int iFormat, numCells, numSpec;
  const char* nameFile = "inputs/dust_density.inp";
  char line[30];
  FILE* densityFile = fopen(nameFile,"r");
  if (densityFile == NULL){
    printf("Failed to open dust_density.inp. Maybe don't exist\n");
    exit(1);
  }else{
    fscanf(densityFile, "%d\n", &iFormat);
    fscanf(densityFile, "%d\n", &numCells);
    fscanf(densityFile, "%d\n", &numSpec);

    if ( (grid->nCoord[0]*grid->nCoord[1]*grid->nCoord[2]) != numCells){
      printf("ERROR: numCells is not equal to nx * ny * nz in amr_grid.inp\n");
      exit(1);
    }

    dustDensity = allocateMemoryToDustDensity(iFormat, numCells, numSpec, grid);
    int count, iSpec,k,j,i;
    for (iSpec = 0 ; iSpec<numSpec ; iSpec++){
      count = 0;
      for (k=0 ; k<grid->nCoord[2] ; k++){
        for (j=0 ; j<grid->nCoord[1] ; j++){
          for (i=0 ; i<grid->nCoord[0] ; i++){
            fgets(line, 30, densityFile);
            //printf("%s", line);
            dustDensity->densities[iSpec][k][j][i] = atof(line);
            //printf(" %10.10lg\n\n",dustDensity->densities[iSpec][k][j][i]);
            count++;
          }
        }
      }
      if (count != numCells){
        printf("Amount of cells is unequal to numCells in dust_density.inp\n");
        exit(1);
      }
    }
    fclose(densityFile);
  }
  return dustDensity;
}

__host__ DustTemperature* setUpDustTemperature(DustDensity* dustDensity, Grid* grid){
  printf("Set up dustTemperature...\n");
  int numSpec = dustDensity->numSpec;
  DustTemperature* dustTemperature = allocateMemoryToDustTemperature(numSpec, grid);
  inicializeDustTemperature(dustTemperature, dustDensity);
  return dustTemperature;
}
