#include <cstdint>
#include <cstdio>
#include <iostream>
#include <stdexcept>

template <class T>
void add(T* a, T* b, T* res, int N);

template <class T>
class Matrix {
  uint32_t M, N;
  T* data;

public:
  Matrix(int M, int N): N(N), M(M) {
    data = new T[M*N];
  }
  
  Matrix(T* data, int N, int M): data(data), N(N), M(M) {}

  ~Matrix() {
    delete[] data;
  }
  
  Matrix transpose() {
    Matrix transposedMatrix(N, M);
    
//    optimize this later with gpu
    for (uint32_t i = 0; i < M; i++) {
      for (uint32_t j = 0; j < N; j++) {
        transposedMatrix.set(j, i) = get(i, j);
      }
    }
    
    return transposedMatrix;
  }
  
  T get(int i, int j) {
    if (i >= M || i < 0 || j >= N || j < 0 )
      throw std::invalid_argument("out of bound error");
    return data[i + j*M];
  }
  
  T* getPointer() {
    return data;
  }
  
  T& set(int i, int j) {
    if (i >= M || i < 0 || j >= N || j < 0 )
      throw std::invalid_argument("out of bound error");
    return data[i + j*M];
  }
  
  Matrix operator+(const Matrix<T>& other) {
    if (M != other.M || N != other.N)
      throw std::invalid_argument("matrix doesn't have same dimension");
    
    Matrix<T> resMat(M, N);
    
    T *dev_a, *dev_b, *dev_res;  
    
    cudaMalloc((void**)&dev_a, M*N * sizeof(T));
    cudaMalloc((void**)&dev_b, M*N * sizeof(T));
    cudaMalloc((void**)&dev_res, M*N * sizeof(T));
    
    cudaMemcpy(dev_a, getPointer(), M*N * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, other.getPointer(), M*N * sizeof(T), cudaMemcpyHostToDevice);
    
    add<<<128,128>>>(dev_a, dev_b, dev_res, N);
    
    cudaMemcpy(resMat.getPointer(), dev_res, M*N * sizeof(T), cudaMemcpyDeviceToHost);
    
    cudaFree(dev_a); cudaFree(dev_b); cudaFree(dev_res);
    
    return resMat;
  }
  
  void print() {
    std::printf("%d, %d \n", M, N);
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        std::cout << get(i, j) << " ";
      }
      std::printf("\n");
    }
  }
};

//int main() {
//  Matrix<float> mat(10, 5);
//  
//  mat.set(3, 0) = 3.0;
//  mat.set(4, 1) = 4.0;
//  mat.set(5, 2) = 5.0;
//  
//  mat.print();
//  std::printf("\n\n");
//  mat.transpose().print();
//  
//  return 0;
//}