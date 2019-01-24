#include <math.h>
#include "global_functions.cuh"
#include "structs.cuh"
#include <unistd.h>

SimulationParameters* parametersTransferToDevice(SimulationParameters h_params){
	SimulationParameters* d_params;
	cudaMalloc((void**)&(d_params), sizeof(SimulationParameters) );
	cudaMemcpy(d_params, &h_params, sizeof(SimulationParameters), cudaMemcpyHostToDevice);
	return d_params;
}

SimulationParameters readInputParameters(int argc, char **argv){
	SimulationParameters param;
	param.nTemp = 1000.0;
	param.temp0 = 0.01;
	param.temp1 = 100000.0;
	param.numPhotons = 0;
  param.numParallelPhotons = 1024*50;
  param.blockSize = 0;
  param.scatteringMode = -1;
	opterr = 0;
	int d;

	while ((d = getopt (argc, argv, "N:n:t:i:f:s:b:")) != -1){
		switch (d){
			case 'N':
				sscanf(optarg, "%d", &param.numPhotons);
				break;
			case 'n':
				sscanf(optarg, "%d", &param.numParallelPhotons);
				break;
			case 't':
				sscanf(optarg, "%d", &param.nTemp);
				break;
			case 'i':
				sscanf(optarg, "%d", &param.temp0);
				break;
			case 'f':
        sscanf(optarg, "%d", &param.temp1);
				break;
      case 's':
        sscanf(optarg, "%d", &param.scatteringMode);
        break;
      case 'b':
        sscanf(optarg, "%d", &param.blockSize);
        break;

			case '?':
				if (optopt == 'N')
					fprintf(stderr, "Opcion -%c requiere un argumento\n", optopt);
				else if (optopt == 'n')
					fprintf(stderr, "Opcion -%c requiere un argumento\n", optopt);
				else if (optopt == 't')
					fprintf(stderr, "Opcion -%c requiere un argumento\n", optopt);
				else if (optopt == 'i')
					fprintf(stderr, "Opcion -%c requiere un argumento\n", optopt);
				else if (optopt == 'f')
					fprintf(stderr, "Opcion -%c requiere un argumento\n", optopt);
				else if (optopt == 's')
					fprintf(stderr, "Opcion -%c requiere un argumento\n", optopt);
				else
					fprintf(stderr,
							"Opcion con caracter desconocido '\\x%x' \n",
							optopt);
				param.statusCode=-1;
				printf("Error con parametros de entrada\n");
				return param;
			default:
				printf("Error con parametros de entrada\n");
				abort ();
		}
	}
	param.statusCode=0;
	return param;
}

void checkSimulationParameters(SimulationParameters param){
  //blockSize
  bool blockIsCorrect=false;
  int sizePosibilities[9] = {2,4,8,16,32,64,128,256,512};
  for (int i=0 ; i<9 ; i++){
    if (param.blockSize == sizePosibilities[i]){
      blockIsCorrect = true;
    }
  }
  if (!blockIsCorrect){
    printf("Error. Block Size is incorrect. Must be 2,4,8,16,32,64,128,256 or 512\n");
    exit(0);
  }

  //numPhotons. Must be multiple of 1024
  if (param.numPhotons%1024 != 0){
    printf("Error. Number of photons denied. Must be multiple of 1024\n");
    exit(0);
  }

  //numParallelPhotons
  if (param.numParallelPhotons > param.numPhotons){
    printf("Error. maxParallelPhotons must be less or equal than number of photons\n");
    exit(0);
  }
  if (param.numPhotons%param.numParallelPhotons != 0){
    printf("Error. maxParallelPhotons must be divisor of number of photons\n");
    exit(0);
  }

  //scatteringMode
  if (param.scatteringMode < 0 || param.scatteringMode > 2 ){
    printf("Error. Scattering mode must be 0,1 or 2\n");
    exit(0);
  }
}

double blackbodyPlanck(double temp, double freq){
  double explargest = 709.78271;
  double bPlanck;

  if (temp == 0){
    bPlanck = 0;
  }else{
    double xx = 4.7989E-11 * freq / temp;
    if (xx > explargest){
      bPlanck = 0;
    }else{
      bPlanck = 1.47455E-47 * freq*freq*freq / (exp(xx)-1.0) + 1E-290;
    }
  }
  return bPlanck;
}
__host__ __device__ void huntFloat(float* xx, int n, double x, int* jlo){
  unsigned int jm, jhi, inc;
  int ascnd;

  ascnd = (xx[n-1] >= xx[0]);
  if ((*jlo <= -1) || (*jlo > n-1)){
    *jlo = -1;
    jhi = n;
  }else{
    inc = 1;
    if (x >= xx[*jlo] == ascnd){
      if (*jlo == n-1){
        return;
      }
      jhi = *jlo+1;
      while ((x >= xx[jhi]) == ascnd){
        *jlo = jhi;
        inc += inc;
        jhi = *jlo+inc;
        if (jhi > n-1){
          jhi=n;
          break;
        }
      }
    }else{
      if (*jlo == 0){
        *jlo = -1;
        return;
      }
      jhi = *jlo-1;
      while ((x < xx[*jlo]) == ascnd){
        jhi = *jlo;
        inc <<=1;
        if (inc >= jhi){
          *jlo = -1;
          break;
        }else{
          *jlo = jhi-inc;
        }
      }
    }
  }

  while (jhi - *jlo != 1){
    jm = (jhi + *jlo) / 2;
    if ((x > xx[jm]) == ascnd){
      *jlo = jm;
    }else{
      jhi = jm;
    }
  }

  if (x == xx[n-1]){
    *jlo = n-2;
  }

  if (x == xx[0]){
    *jlo = 0;
  }
}

__host__ __device__ void huntDouble(double* xx, int n, double x, int* jlo){
  unsigned int jm, jhi, inc;
  int ascnd;

  ascnd = (xx[n-1] >= xx[0]);
  if ((*jlo <= -1) || (*jlo > n-1)){
    *jlo = -1;
    jhi = n;
  }else{
    inc = 1;
    if (x >= xx[*jlo] == ascnd){
      if (*jlo == n-1){
        return;
      }
      jhi = *jlo+1;
      while ((x >= xx[jhi]) == ascnd){
        *jlo = jhi;
        inc += inc;
        jhi = *jlo+inc;
        if (jhi > n-1){
          jhi=n;
          break;
        }
      }
    }else{
      if (*jlo == 0){
        *jlo = -1;
        return;
      }
      jhi = *jlo-1;
      while ((x < xx[*jlo]) == ascnd){
        jhi = *jlo;
        inc <<=1;
        if (inc >= jhi){
          *jlo = -1;
          break;
        }else{
          *jlo = jhi-inc;
        }
      }
    }
  }

  while (jhi - *jlo != 1){
    jm = (jhi + *jlo) / 2;
    if ((x > xx[jm]) == ascnd){
      *jlo = jm;
    }else{
      jhi = jm;
    }
  }

  if (x == xx[n-1]){
    *jlo = n-2;
  }

  if (x == xx[0]){
    *jlo = 0;
  }
}

double* remapFunction(int nOld, double* xOld, double* fOld, int nNew, double* xNew, int elow){
  double* fNew = (double*)malloc(sizeof(double)*nNew);
  int value = 400;
  int *iold = &value;
  double eps;
  int sgnOld;
  int sgnNew;
  int startOld, endOld;
  int startNew;
  int i;

  //verify if x is increasing
  if (xOld[nOld-1] >= xOld[0]){
    sgnOld = 1;
    startOld = 0;
    endOld = nOld-1;
  }else{
     // decreasing
     sgnOld=-1;
     startOld= nOld-1;
     endOld=0;
  }

  if (xNew[nNew-1] >= xNew[0]) {
    sgnNew = 1;
    startNew = 0;
  }else{
    sgnNew = -1;
    startNew = nNew-1;
  }
  //printf("nOld=%d\n",nOld);
  //printf("startOld=%d\n",startOld);
  for (i=0 ; i<nNew ; i++){
    //printf("xNew= %10.10lg, xOld= %10.10lg\n",xNew[startNew], xOld[startOld]);
    //Lower than lowest boundary
    if (xNew[startNew] < xOld[startOld]){
      if (elow == 1){
        fNew[startNew] = fOld[startOld];
      }else if (elow == 2){
        if ((fOld[startOld+sgnOld]>0) && (fOld[startOld]>0)){
          double x = fOld[startOld+sgnOld]/fOld[startOld];
          double y = (log(xNew[startNew]) - log(xOld[startOld])) / (log(xOld[startOld+sgnOld]) - log(xOld[startOld]));
          fNew[startNew] = fOld[startOld] * pow(x,y);
        }
      }
    }
    //higher than highest boundary
    else if (xNew[startNew] > xOld[endOld]){
      fNew[startNew] = fOld[endOld];
    }
    else{
      //within the domain of lambda in frquencies
      huntDouble(xOld, nOld, xNew[startNew], iold);
      //printf("iold=%d\n",*iold);
      if (xNew[startNew] == xOld[0]){
        fNew[startNew] = fOld[0];
      }else if (xNew[startNew] == xOld[nOld-1]){
        fNew[startNew] = fOld[nOld-1];
      }else{
        if ((*iold <= -1) || (*iold >= nOld-1)){
          printf("ERROR in remap function\n");
          exit(1);
        }
        eps = (xNew[startNew]-xOld[*iold]) / (xOld[*iold+1]-xOld[*iold]);
        if ((fOld[*iold] > 0) && (fOld[*iold+1]>0)){
          fNew[startNew] = exp((1-eps) * log(fOld[*iold]) + eps * log(fOld[*iold+1]));
        }else{
          fNew[startNew] = (1-eps) * fOld[*iold] + eps*fOld[*iold+1];
        }
      }
    }
    //printf("new: %10.10lg\n",fNew[startNew]);
    startNew += sgnNew;
  }
  return fNew;

}
