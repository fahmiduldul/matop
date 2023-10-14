#include <algorithm>
#include <stdexcept>
#include <string>
#include "handler.cuh"

template<class T>
__global__ void dev_add(T* a, T* b, T* res, int N) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;

  while (idx < N) {
    res[idx] = a[idx] + b[idx];
    idx += blockDim.x * gridDim.x;
  }
};

template<class T>
__global__ void dev_subtract(T* a, T* b, T* res, int N) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;

  while (idx < N) {
    res[idx] = a[idx] - b[idx];
    idx += blockDim.x * gridDim.x;
  }
};

template<class T>
__global__ void dev_multiply(T* a, T* b, T* res, int N) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;

  while (idx < N) {
    res[idx] = a[idx] * b[idx];
    idx += blockDim.x * gridDim.x;
  }
};

template<class T>
void cuda_simple_op(T* a, T* b, T* res, int N, std::string& op) {
  T *dev_a, *dev_b, *dev_res;
  
  HANDLE_ERROR( cudaMalloc((void**)&dev_a, N * sizeof(T)) );
  HANDLE_ERROR( cudaMalloc((void**)&dev_b, N * sizeof(T)) );
  HANDLE_ERROR( cudaMalloc((void**)&dev_res, N * sizeof(T)) );

  HANDLE_ERROR( cudaMemcpy(dev_a, a, N * sizeof(T), cudaMemcpyHostToDevice) );
  HANDLE_ERROR( cudaMemcpy(dev_b, b, N * sizeof(T), cudaMemcpyHostToDevice) );

  // TODO: improve block and thread dimension to be depend on device capacity
  int nBlock = 64; int nThread = 256;
  if(op.compare("add") == 0) {
    dev_add<<<nBlock,nThread>>>(dev_a, dev_b, dev_res, N);
  } else if (op.compare("substract") == 0) {
    dev_subtract<<<nBlock,nThread>>>(dev_a, dev_b, dev_res, N);
  } else if (op.compare("multiply") == 0) {
    dev_multiply<<<nBlock,nThread>>>(dev_a, dev_b, dev_res, N);
  } else {
    throw std::invalid_argument("\"add\", \"substract\" & \"multiply\" are the only valid operation values")
  }

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

    // TODO: improve block and thread dimension to be depend on device capacity
    int nBlock = 64; int nThread = 256;
    dev_transpose<<<nBlock, nThread>>>(dev_input, dev_result, M, N);

    HANDLE_ERROR( cudaMemcpy(res, dev_result, M * N * sizeof(T), cudaMemcpyDeviceToHost) );

    cudaFree(dev_input); cudaFree(dev_result);
};

template<class T>
void dev_sum(T* a, T* partial_res, int M) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int it = tid;

    T temp = 0;
    while(it < M) {
      temp += a[tid];
      it += blockDim.x * gridDim.x;
    }

    partial_res[tid] = temp;
};

template<class T>
T cuda_dot_product(T* a, T* b, int M) {
    T *dev_a, *dev_b, *dev_mult_res, *dev_partial_res;

    // TODO: improve block and thread dimension to be depend on device capacity
    int threadPerBlock = std::min(256, M);
    int nBlock = std::max(M / threadPerBlock + 1, 64);

    HANDLE_ERROR( cudaMalloc((void**)&dev_a, M * sizeof(T)) );
    HANDLE_ERROR( cudaMalloc((void**)&dev_b, M * sizeof(T)) );
    HANDLE_ERROR( cudaMalloc((void**)&dev_mult_res, M * sizeof(T)) );

    HANDLE_ERROR( cudaMemcpy(dev_a, a, N * sizeof(T), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(dev_b, b, N * sizeof(T), cudaMemcpyHostToDevice) );

    dev_multiply<<<nBlock, threadPerBlock>>>(dev_a, dev_b, dev_mult_res, M, "multiply");

    cudaFree(dev_a); cudaFree(dev_b);

    HANDLE_ERROR( cudaMalloc((void**)&dev_partial_result, threadPerBlock * nBlock * sizeof(T)) );
    while(nBlock > 0) {
      dev_sum<<<nBlock, threadPerBlock>>>(dev_mult_res, dev_partial_result, M);

      cudaFree(dev_mult_res);
      dev_mult_res = dev_partial_result;
      nBlock /= 4;
      HANDLE_ERROR( cudaMalloc((void**)&dev_partial_result, threadPerBlock * nBlock * sizeof(T)) );
    }

    T tempResArr[threadPerBlock];
    HANDLE_ERROR( cudaMemcpy(tempResArr, dev_partial_res, threadPerBlock * sizeof(T), cudaMemcpyDeviceToHost) );
    cudaFree(dev_partial_res);

    T tempRes = 0;
    for (int i = 0; i < threadPerBlock; i++) {
      tempRes += tempResArr[i];
    }
    
    return tempRes;
};


void util() {
  // this function is needed so that the compiler compiles those functions above in the object file

  double* a;
  cuda_simple_op(a,a,a,1, "add");
  cuda_transpose(a,a,1,1);
  
  long* b;
  cuda_simple_op(b,b,b,1, "add");
  cuda_transpose(b,b,1,1);
}
