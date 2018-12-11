#include <math.h>
/*
--------------------------------------------------------------------------
                THE BLACKBODY PLANCK FUNCTION B_nu(T)

     This function computes the Blackbody function

                    2 h nu^3 / c^2
        B_nu(T)  = ------------------    [ erg / cm^2 s ster Hz ]
                   exp(h nu / kT) - 1

     ARGUMENTS:
        freq    [Hz]            = Frequency
        temp  [K]             = Temperature
*/
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

__host__ __device__ void huntCPU(double* xx, int n, double x, int* jlo){
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

double* remapFunction(int nOld, double* xOld, double* fOld, int nNew, float* xNew, int elow){
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
      huntCPU(xOld, nOld, xNew[startNew], iold);
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

__device__ double doubleAtomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}

__global__ void huntHelper(double* xx, double x, int* index, int offset){
  unsigned int tid = offset + threadIdx.x + blockDim.x * blockIdx.x;
  if ( x >= xx[tid] && x < xx[tid+1]){
    //printf("thread=%d\n",tid);
    *index = tid;
  }
}

__device__ void huntGPU(double* xx, int n, double x, int* index){
  if (x == xx[n-1]){
    *index = n-2;
    //newIndex = ;
    //printf("index: %d\n",*index);
    return;
  }

  if (x < xx[0]){
    *index = -1;
    //newIndex = -1;
    //printf("index: %d\n",*index);
    return;
  }
  if (x > xx[n-1]){
    *index = n;
    //newIndex = n;
    //printf("index: %d\n",*index);
    return;
  }

  int threadsPerBlock = 256;
  int numBlocks=1;
  int restantThreads=0;
  int offset;
  int threadToLaunch=n-1;
  if ( (n-1) >= threadsPerBlock){
    numBlocks = (n-1)/threadsPerBlock;
    restantThreads = (n-1) - numBlocks*threadsPerBlock;
    offset = numBlocks*threadsPerBlock;
    threadToLaunch = threadsPerBlock;
  }
  //printf("numThreads to Launch:%d\n",n-1);
  huntHelper<<<numBlocks, threadToLaunch>>>(xx,x,index,0);
  if (restantThreads>0){
    huntHelper<<<1, restantThreads>>>(xx,x,index,offset);
  }
  cudaDeviceSynchronize();
  //return index;
  //newIndex = *index;
  //printf("index: %d\n",*index);
  //return *index;
}
