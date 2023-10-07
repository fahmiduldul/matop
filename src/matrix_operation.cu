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
void cuda_add(T* a, T* b, T* res, int N) {
  T *dev_a, *dev_b, *dev_res;
  
  HANDLE_ERROR( cudaMalloc((void**)&dev_a, N * sizeof(T)) );
  HANDLE_ERROR( cudaMalloc((void**)&dev_b, N * sizeof(T)) );
  HANDLE_ERROR( cudaMalloc((void**)&dev_res, N * sizeof(T)) );

  HANDLE_ERROR( cudaMemcpy(dev_a, a, N * sizeof(T), cudaMemcpyHostToDevice) );
  HANDLE_ERROR( cudaMemcpy(dev_b, b, N * sizeof(T), cudaMemcpyHostToDevice) );

  dev_add<<<128,128>>>(dev_a, dev_b, dev_res, N);

  HANDLE_ERROR( cudaMemcpy(res, dev_res, N * sizeof(T), cudaMemcpyDeviceToHost) );

  cudaFree(dev_a); cudaFree(dev_b); cudaFree(dev_res);
}


template<class T>
__global__ void dev_transpose(T* input, T* res, int M, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    while(idx < M*N) {
      int i = idx % M;
      int j = idx / N;

      res[N*i + j] = input[M*j + i];

      idx += blockDim.x * gridDim.x;
    }
};

template<class T>
void cuda_transpose(T* input, T* res, int M, int N) {
    T *dev_input, *dev_result;

    HANDLE_ERROR( cudaMalloc((void**)&dev_input, M * N * sizeof(T)) );
    HANDLE_ERROR( cudaMalloc((void**)&dev_result, M * N * sizeof(T)) );

    HANDLE_ERROR( cudaMemcpy(dev_input, input, M * N * sizeof(T), cudaMemcpyHostToDevice) );

    dev_transpose<<<128,128>>>(dev_input, dev_result, M, N);

    HANDLE_ERROR( cudaMemcpy(res, dev_result, M * N * sizeof(T), cudaMemcpyDeviceToHost) );

    cudaFree(dev_input); cudaFree(dev_result);
};



void util() {
  // this function is needed so that the compiler compiles those functions above in the object file

  double* a;
  cuda_add(a,a,a,1);
  cuda_transpose(a,a,1,1);
  
  long* b;
  cuda_add(b,b,b,1);
  cuda_transpose(b,b,1,1);
}
