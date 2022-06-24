#include <iostream>
#include "cuda_runtime.h"  

// 矩阵乘法
// m*l l*n
__global__ void matrix_mul(float* x, float * y, float* z, int m, int n, int l)
{
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;
  const int row = idx / m;
  const int col = idx % m;
  if(row < m && col < n) {
    for(int i = 0; i < l; i++) {
      z[row*n+ col] += x[row*l + i] * y[i*n + col];
    }
  }
}

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

  // 申请device内存
  float *d_x, *d_y, *d_z;
  cudaMalloc((void**)&d_x, M*L*sizeof(float));
  cudaMalloc((void**)&d_y, L*N*sizeof(float));
  cudaMalloc((void**)&d_z, M*N*sizeof(float));

  // 将host数据拷贝到device
  cudaMemcpy((void*)d_x, (void*)x, M*L*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy((void*)d_y, (void*)y, L*N*sizeof(float), cudaMemcpyHostToDevice);

  // 定义kernel的执行配置
  
  dim3 threads(1024);
  dim3 blocks(M*N-1024+1/1024);
  matrix_mul <<<blocks, threads>>>(d_x, d_y, d_z, M, N, L);

  // 将device得到的结果拷贝到host
  cudaMemcpy((void*)z, (void*)d_z, M*N*sizeof(float), cudaMemcpyDeviceToHost);

  // 输出前10个数值
  for(int i = 0; i < 10; i++) {
    std::cout << z[i] << " ";
  }
  std::cout << std::endl;

  std::cout << "Done!" << std::endl;

  // 释放device内存
  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_z);
  // 释放host内存
  free(x);
  free(y);
  free(z);

  return 0;
}
