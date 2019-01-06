#include "dust.cuh"

__global__ void printEner(DustTemperature* dustTemperature, DustDensity* dustDensity){
  int contador=0;
  for (int iSpec=0 ; iSpec<dustDensity->numSpec ; iSpec++){
    for (int k=0 ; k<dustDensity->nz ; k++){
      for (int j=0 ; j<dustDensity->ny ; j++){
        for (int i=0 ; i<dustDensity->nx ; i++){
          //if (contador < 10){
          if ( dustTemperature->cumulEner[iSpec][k][j][i] == 0){
            printf("ener %d,%d,%d: %10.10lg\n", i,j,k, dustTemperature->cumulEner[iSpec][k][j][i]);
          }
        }
      }
    }
  }
}

__global__ void printTemp(DustTemperature* dustTemperature, DustDensity* dustDensity){
  int contador=0;
  for (int iSpec=0 ; iSpec<dustDensity->numSpec ; iSpec++){
    for (int k=0 ; k<dustDensity->nz ; k++){
      for (int j=0 ; j<dustDensity->ny ; j++){
        for (int i=0 ; i<dustDensity->nx ; i++){
          if (contador < 7500){
            printf("temp %d: %10.10lg\n", contador, dustTemperature->temperatures[iSpec][k][j][i]);
            contador++;
          }

        }
      }
    }
  }
}

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


/*
double*** transferDoublePointer(){
  printf("transferDoublePointer\n");
  int N = 3;
  int N2 = 4;
  int N3 = 5;
  double*** values_device;
  double*** support3;
  double** support2;
  double*** values;
  values =(double***) malloc(sizeof(double**)*N);
  for (int i=0;i<N;i++){
    values[i] =(double**) malloc(sizeof(double*)*N2);
    for (int j=0;j<N2;j++){
      values[i][j] =(double*) malloc(sizeof(double)*N3);
      for (int k=0 ; k<N3;k++){
        values[i][j][k] = i+j+k+1;
        printf("%1.1lf ",values[i][j][k]);
      }
      printf("\n");
    }
    printf("\n");
  }
  printf("cudaMalloc values_device\n");
  cudaMalloc((void **) &values_device,N*sizeof(double**));
  support3 =(double***) malloc(sizeof(double**)*N);
  for (int i=0;i<N;i++){
    printf("cudaMalloc support3-%d\n",i);
    cudaMalloc((void**) &support3[i],sizeof(double*)*N2);
    support2 =(double**) malloc(sizeof(double*)*N2);
    for (int j=0;j<N2;j++){
      printf("cudaMalloc support2-%d\n",j);
      cudaMalloc((void**) &support2[j],sizeof(double)*N3);
      cudaMemcpy(support2[j],values[i][j],N3*sizeof(double),cudaMemcpyHostToDevice);
    }
    cudaMemcpy(support3[i],support2,sizeof(double*)*N2,cudaMemcpyHostToDevice);
    free(support2);
  }
  cudaMemcpy(values_device,support3,sizeof(double**)*N,cudaMemcpyHostToDevice);
  free(support3);

  //DOUBLE** POINTER
  cudaMalloc((void **) &values_device,N*sizeof(double*));
  support =(double**) malloc(sizeof(double*)*N);

  for (int i=0;i<N;++i){
    printf("cudaMalloc support%d\n",i);
    cudaMalloc((void**) &support[i],sizeof(double)*N2);
    printf("cudaMemcpy to values_device%d\n",i);
    //cudaMemcpy(&values_device[i], &support[i], sizeof(double)*N2, cudaMemcpyDeviceToDevice);
    //printf("in cudaMemcpy\n");
    printf("cudaMemcpy values%d\n",i);
    cudaMemcpy(support[i],values[i],N2*sizeof(double),cudaMemcpyHostToDevice);
  }
  cudaMemcpy(values_device,support,sizeof(double*)*N,cudaMemcpyHostToDevice);
  printf("exit transferDoublePointer...\n");*
  return values_device;


}*/
/*
__global__ void printfDoubles(double*** d_values){
  printf("IN DOUBLES\n");
  int i = threadIdx.x;
  //d_values[0][0] = 5;
  //d_values[1][0] = 6;
  printf("threadIdx%d\n",i);
  for (int i=0 ; i<3 ; i++){
    for (int j=0 ; j<4 ; j++){
      for (int k=0 ; k<5 ; k++){
        printf("%1.1lf ",d_values[i][j][k]);
      }
      printf("\n");
    }
    printf("\n");
  }
}*/
/*
__global__ void lookDensities(DustDensity* dustDensity){
  printf("IN DENSITIES\n");
  int i = threadIdx.x;
  printf("threadIdx%d\n",i);
  printf("densities[0][0][%d][0] = %lg\n",i,dustDensity->densities[0][i][0][0]);
  printf("densities[0][0][%d][1] = %lg\n",i,dustDensity->densities[0][i][0][1]);
  printf("numSpec = %d\n",dustDensity->numSpec);
  printf("numCells= %d\n",dustDensity->numCells);

}*/

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
