template<class T>
__global__ void add(T* a, T* b, T* res, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    
    while (idx < N) {
      res[idx] = a[idx] + b[idx];
      idx += blockDim.x * gridDim.x;
    }
};
