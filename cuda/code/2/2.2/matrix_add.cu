// 矩阵加法

// m*n + m*n
__global__ void matrix_add(float* matA, float* matB, float* matC, int m, int n) {
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    int idx = ix + iy*n; // 注意这里， iy才是表示行， ix表示列
    if(ix < m && iy < n) {
        matC[idx] = matA[idx] + matB[idx];
    }
}

#include <iostream>
int main()
{
    int M = 512;
    int N = 512;

    // 申请host内存
    float *x, *y, *z;
    x = (float*)malloc(sizeof(float)*M*N);
    y = (float*)malloc(sizeof(float)*M*N);
    z = (float*)malloc(sizeof(float)*M*N);

    // 初始化数据
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < N; j++) {
            x[i*N + j] = i * 1.0 + j;
            y[i*N + j] = i * 1.0 + j;
            z[i*N + j] = 0.0;
        }
    }

    // 申请device内存
    float *d_x, *d_y, *d_z;
    cudaMalloc((void**)&d_x, sizeof(float)*M*N);
    cudaMalloc((void**)&d_y, sizeof(float)*M*N);
    cudaMalloc((void**)&d_z, sizeof(float)*M*N);

    // 将host数据拷贝到device
    cudaMemcpy((void*)d_x, (void*)x, sizeof(float)*M*N, cudaMemcpyHostToDevice);
    cudaMemcpy((void*)d_y, (void*)y, sizeof(float)*M*N, cudaMemcpyHostToDevice);

    // 定义kernel的执行配置
    // 2d block and 2d grid
    // dim3 blockSize(32, 32);
    // dim3 gridSize((M + blockSize.x - 1) / blockSize.x, (N + blockSize.x - 1) / blockSize.x);

    1d block and 1d grid
    dim3 blockSize(32);
    dim3 gridSize((M*N + blockSize.x - 1) / blockSize.x);

    // 2d block and 1d grid
    dim3 blockSize(32);
    dim3 gridSize((M + blockSize.x - 1) / blockSize.x, N);

    // 执行kernel
    matrix_add<<< gridSize, blockSize >>>(d_x, d_y, d_z, M, N);
    printf("执行kernel： matrix<<<(%d, %d), (%d, %d)>>>\n", gridSize.x, gridSize.y, blockSize.x, blockSize.y);

    // 将device得到的结果拷贝到host
    cudaMemcpy((void*)z, (void*)d_z, sizeof(float)*M*N, cudaMemcpyDeviceToHost);

    // 检查执行结果
    std::cout << "第一行前十个结果为: " << std::endl;
    for (int i = 0; i < 10; i++)
       std::cout << z[i] << " ";
    std::cout << std::endl;
    
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
