#pragma once

#include <cstdint>
#include <cstdio>
#include <stdexcept>

template<class T>
void add(T* a, T* b, T* res, int N);

class DoubleMatrix {
  uint32_t M, N;
  double* data;

public:
  DoubleMatrix(int M, int N): N(N), M(M) {
    data = new double[M*N];
  }
  
  DoubleMatrix(double* data, int N, int M): data(data), N(N), M(M) {}

  ~DoubleMatrix() {
    delete[] data;
  }
  
  DoubleMatrix transpose() {
    DoubleMatrix transposedMatrix(N, M);
    
//    optimize this later with gpu
    for (uint32_t i = 0; i < M; i++) {
      for (uint32_t j = 0; j < N; j++) {
        transposedMatrix.set(j, i) = get(i, j);
      }
    }
    
    return transposedMatrix;
  }
  
  double get(int i, int j) const {
    if (i >= M || i < 0 || j >= N || j < 0 )
      throw std::invalid_argument("out of bound error");
    return data[i + j*M];
  }
  
  double* getPointer() const {
    return data;
  }
  
  double& set(int i, int j) {
    if (i >= M || i < 0 || j >= N || j < 0 )
      throw std::invalid_argument("out of bound error");
    return data[i + j*M];
  }
  
  DoubleMatrix operator+(const DoubleMatrix& other) {
    if (M != other.M || N != other.N)
      throw std::invalid_argument("matrix doesn't have same dimension");
    
    DoubleMatrix resMat(M, N);
    
    add(getPointer(), other.getPointer(), resMat.getPointer(), M * N);

    
    return resMat;
  }
  
  void print() {
    std::printf("%d, %d \n", M, N);
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        std::printf("%.2f ", get(i, j));
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