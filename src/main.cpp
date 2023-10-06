#include "matrix.h"


int main() {
  DoubleMatrix matA(5, 3);
  DoubleMatrix matB(5, 3);
  
  matA.set(3, 0) = 3.0;
  matA.set(4, 1) = 4.0;
  matA.set(1, 2) = 5.0;
  
  matB.set(3,1) = 9.0;
  
  DoubleMatrix matRes = matA + matB;
  
  matA.print();
  matB.print();
  
  matRes.print();
  
  return 0;
}