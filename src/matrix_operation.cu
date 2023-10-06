#include "handler.h"

template<class T>
__global__ void dev_add(T* a, T* b, T* res, int N) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;

  while (idx < N) {
    res[idx] = a[idx] + b[idx];
    idx += blockDim.x * gridDim.x;
  }
};

template<class T>
void add(T* a, T* b, T* res, int N) {
  T *dev_a, *dev_b, *dev_res;
  
  HANDLE_ERROR( cudaMalloc((void**)&dev_a, N * sizeof(double)) );
  HANDLE_ERROR( cudaMalloc((void**)&dev_b, N * sizeof(double)) );
  HANDLE_ERROR( cudaMalloc((void**)&dev_res, N * sizeof(double)) );

  HANDLE_ERROR( cudaMemcpy(dev_a, a, N * sizeof(double), cudaMemcpyHostToDevice) );
  HANDLE_ERROR( cudaMemcpy(dev_b, b, N * sizeof(double), cudaMemcpyHostToDevice) );

  HANDLE_ERROR( dev_add<<<128,128>>>(dev_a, dev_b, dev_res, N) );

  HANDLE_ERROR( cudaMemcpy(res, dev_res, N * sizeof(double), cudaMemcpyDeviceToHost) );

  cudaFree(dev_a); cudaFree(dev_b); cudaFree(dev_res);
}

void util() {
  double* a;
  add(a,a,a,1);
  
  long* b;
  add(b,b,b,1);
}
