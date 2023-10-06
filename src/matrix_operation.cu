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
  
  cudaMalloc((void**)&dev_a, N * sizeof(double));
  cudaMalloc((void**)&dev_b, N * sizeof(double));
  cudaMalloc((void**)&dev_res, N * sizeof(double));

  cudaMemcpy(dev_a, a, N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_b, b, N * sizeof(double), cudaMemcpyHostToDevice);

  dev_add<<<128,128>>>(dev_a, dev_b, dev_res, N);

  cudaMemcpy(res, dev_res, N * sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(dev_a); cudaFree(dev_b); cudaFree(dev_res);
}