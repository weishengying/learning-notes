// 矩阵乘法:CPU下转置矩阵乘法
// m*l l*n
#include <stdlib.h>
void matrix_mul(float* x, float * y, float* z, int m, int n, int l)
{
  float* transposed = (float*)malloc(l*n*sizeof(float));
  // l*n --> n*l
  for(int i = 0; i < n; i++) {
    for(int j = 0; j < l; j++) {
      transposed[i*l + j] = y[j*n + i];
    }
  }

  //m*l n*l
  for(int i = 0; i < m; i++) {
    for(int j = 0; j < n; j++) {
      for(int k = 0; k < l; k++) {
        z[i*n + j] += x[i*l + k] * y[j*l + k];
      }
    }
  }

  free(transposed);
}

#include <iostream>
#include <assert.h>
#include <chrono>

int main()
{
  int M = 2048;
  int L = 1024;
  int N = 512;

  // 申请host内存
  float *x = NULL;
  float *y = NULL;
  float *z = NULL;
  x = (float*)malloc(M*L*sizeof(float));
  y = (float*)malloc(L*N*sizeof(float));
  z = (float*)malloc(M*N*sizeof(float));

  if(x == NULL || y == NULL || z == NULL)
    return 0;

  // 初始化数据
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < L; ++j) {
      x[i*L + j] = 1.1;
    }
  }
  for (int i = 0; i < L; ++i) {
    for (int j = 0; j < N; ++j) {
      y[i*N + j] = 1.1;
    }
  }
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      z[i*N + j] = 0;
    }
  }

  // 执行矩阵乘法
  using namespace std::chrono;
  steady_clock::time_point t1 = steady_clock::now();
  matrix_mul(x, y, z, M, N, L);
  steady_clock::time_point t2 = steady_clock::now();

  duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
  std::cout << "It took me " << time_span.count() << " seconds.";
  std::cout << std::endl;

  // 输出前10个数值
  for(int i = 0; i < 10; i++) {
    std::cout << z[i] << " ";
  }
  std::cout << std::endl;

  std::cout << "Done!" << std::endl;

  // 释放host内存
  free(x);
  free(y);
  free(z);

  return 0;
}
