
template<class T>
class Matrix;

int main() {
  Matrix<float> matA(5, 3);
  Matrix<float> matB(5, 3);
  
  matA.set(3, 0) = 3.0;
  matA.set(4, 1) = 4.0;
  matA.set(1, 2) = 5.0;
  
  matB.set(3,1) = 9.0;
  
  Matrix<float> matRes = matA + matB; 
  matRes.print();
  
  return 0;
}