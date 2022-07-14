#include "NvInferPlugin.h"
#include "LeakRelu_Plugin.h"
#include "util.h"

#include <cassert>
#include <iostream>
#include <string>
#include <vector>
#include <memory.h>

using namespace nvinfer1;

__global__ void add(float* x, float* y, float negSlope, int n)
{
    // 获取全局索引
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    // 步长
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
    {
        if(x[i] > 0)
            y[i] = x[i];
        else
            y[i] = negSlope * x[i];
    }
}

int ReluCudaKernel(int batchSize, float negSlope, const void* input, void* output, int nbInputEle) {
    // 定义kernel的执行配置
    dim3 gridSize(1);
    dim3 blockSize(2);
    // 执行kernel
    add <<< gridSize, blockSize >>>((float*)input, (float*)output, negSlope, nbInputEle);
    return 0;
}   

LReLU::LReLU(const void* buffer, size_t length)
{
    const char *d = reinterpret_cast<const char *>(buffer), *a = d;
    mNegSlope = read<float>(d);
    input_dims_ = read<Dims>(d);
    assert(d == a + length);
}

Dims LReLU::getOutputDimensions(int index, const Dims* inputs, int nbInputDims) noexcept
{
    assert(nbInputDims == 1);
    assert(index == 0);
    return inputs[0];
}

void LReLU::configureWithFormat(Dims const* inputDims, int32_t nbInputs, Dims const* outputDims, int32_t nbOutputs,
        DataType type, PluginFormat format, int32_t maxBatchSize) noexcept
{
    assert(nbInputs == 1);
    assert(nbOutputs == 1);
    input_dims_ = inputDims[0];
}

int LReLU::enqueue(int batchSize, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    const void* inputData = inputs[0];
    void* outputData = outputs[0];
    int status = ReluCudaKernel(batchSize, mNegSlope, inputData, outputData, batchSize * ProductOfDims(input_dims_));
    return status;
}

void LReLU::serialize(void* buffer) const noexcept
{
    char *d = reinterpret_cast<char *>(buffer), *a = d;
    write<float>(d, mNegSlope);
    write<Dims>(d, input_dims_);
    assert(d == a + getSerializationSize());
}


IPluginV2* LReLU::clone() const noexcept {
  auto* plugin = new LReLU(mNegSlope, input_dims_);
  return plugin;
}
