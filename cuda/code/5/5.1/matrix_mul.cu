#include <iostream>
#include "cuda_runtime.h"  

// 矩阵乘法:棋盘阵列矩阵乘法
// m*l l*n

const __device__ int threadnx = 16;

__global__ void matrix_mul(float* x, float * y, float* z, int m, int n, int l)
{
  __shared__ float matA[threadnx][threadnx];
  __shared__ float matB[threadnx][threadnx];
  const int tidr = threadIdx.x;
  const int tidc = threadIdx.y;
  const int bidr = blockIdx.x * threadnx;
  const int bidc = blockIdx.y * threadnx;
  double results = 0.0;

  for(int j = 0; j < l; j += threadnx) {
    // if(bidr + tidr < m && tidc + j < l) {
      matA[tidr][tidc] = x[(tidr + bidr) * l + tidc + j];
    // } else {
    //   matA[tidr][tidc] = 0.0;
    // }

    // if(tidr + j < l && bidc + tidc < n) {
      matB[tidr][tidc] = y[(tidr + j) * n + bidc + tidc];
    // } else {
    //   matB[tidr][tidc] = 0.0;
    // }

    __syncthreads();
    
    for(int i = 0; i < threadnx; i++) {
      results += matA[tidr][i] * matB[i][tidc];
    }

    __syncthreads();
  }

  if(tidr + bidr < m && tidc + bidc < n) {
    z[(tidr + bidr)*n + tidc + bidc] = results;
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
  int bx = (M + threadnx - 1) / threadnx;
  int by = (N + threadnx - 1) / threadnx;
  dim3 blocks(bx, by);
  dim3 threads(threadnx, threadnx);

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
