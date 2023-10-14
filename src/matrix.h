#pragma once

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <stdexcept>
#include <string>
#include "matrix_operation.cuh"

template<class T>
class Matrix {
  uint32_t M, N;
  T* data;

public:
  Matrix(int M, int N): N(N), M(M) {
    data = new double[M*N];
  }
  
  Matrix(double* data, int N, int M): data(data), N(N), M(M) {}

  ~Matrix() {
    delete[] data;
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
  
  Matrix operator+(const Matrix& other) {
    if (M != other.M || N != other.N)
      throw std::invalid_argument("matrix doesn't have same dimension");
    
    Matrix resMat(M, N);
    
    cuda_simple_op(getPointer(), other.getPointer(), resMat.getPointer(), M * N, "add");
    
    return resMat;
  }

  Matrix operator-(const Matrix& other) {
    if (M != other.M || N != other.N)
      throw std::invalid_argument("matrix doesn't have same dimension");

    Matrix resMat(M, N);

    cuda_simple_op(getPointer(), other.getPointer(), resMat.getPointer(), M * N, "substract");

    return resMat;
  }

  Matrix operator*(const Matrix& other) {
    // fix this to matrix operation instead of simple operation
    if (M != other.M || N != other.N)
      throw std::invalid_argument("matrix doesn't have same dimension");

    Matrix resMat(M, N);

    cuda_simple_op(getPointer(), other.getPointer(), resMat.getPointer(), M * N, "multiply");

    return resMat;
  }

  Matrix operator*(long num) {
    long numMat[M*N];
    for (unsigned long i = 0; i < M*N; i++) {
      numMat[0] = num;
    }

    Matrix resMat(M, N);

    cuda_simple_op(getPointer(), numMat, resMat.getPointer(), M * N, "multiply");

    return resMat;
  }

  Matrix dot(const Matrix& other) {
    if (M != other.M || N != 1 || other.N != 1)
      throw std::invalid_argument("dot product only support vector with same size");

    return cuda_dot_product(getPointer(), other.getPointer(), M);
  }

  Matrix transpose() {
    Matrix transposedMatrix(N, M);

    cuda_transpose(getPointer(), transposedMatrix.getPointer(), M, N);

    return transposedMatrix;
  }

  void print() {
    std::printf("%d, %d \n", M, N);
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        if constexpr (std::is_same<T, double>::value)
          std::printf("%.2f ", get(i, j));
        else
          std::printf("%ld ", get(i,j));
      }
      std::printf("\n");
    }
  }

  static Matrix createRandom(int M, int N) {
    Matrix res = Matrix(M, N);

    std::srand((unsigned)time(NULL));

    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        res.set(i,j) = (T)(std::rand() * 1000);
      }
    }
    
    return res;
  }
};

// clang++ src/main.cpp src/matrix_operation.cu -o main --cuda-gpu-arch=sm_35 -L/usr/lib/cuda/lib64 -lcudart_static -ldl -lrt -pthread