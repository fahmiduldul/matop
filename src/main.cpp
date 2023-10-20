#include "matrix.hpp"
#include <iostream>


int main() {
  Matrix<double> matA(5, 3);
  Matrix<double> matB(5, 3);
  
  matA.set(3, 0) = 3.0;
  matA.set(4, 1) = 4.0;
  matA.set(1, 2) = 5.0;
  
  matB.set(3,1) = 9.0;
  
  Matrix<double> matRes = matA + matB;
  
  matA.print();
  matB.print();
  
  matRes.print();
  
  std::cout << "----------------------------------------" << std::endl;

  matRes.transpose().print();

  std::cout << "----------------------------------------" << std::endl;

  auto randomMat = Matrix<double>::createRandom(10,4);
  randomMat.print();

  std::cout << "----------------------------------------" << std::endl;
  
  auto mat1 = Matrix<double>::createRandom(10,1);
  auto mat2 = Matrix<double>::createRandom(10,1);

  mat1.print();
  mat2.print();

  double res = mat1.dot(mat2);

  std::cout << res << std::endl;
  
  return 0;
}
