
#include <iostream>
#include <cuda_runtime_api.h>
#include <vector>
#include <assert.h>

#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "util.h"
#include "LeakRelu_Plugin.h"


using namespace nvinfer1;

class Logger : public ILogger
{
public:
    void log(Severity severity, const char* msg) noexcept override
    {
        // suppress info-level messages
        if (severity <= Severity::kERROR)
            std::cout << msg << std::endl;
    }
};

int main() {
  Logger logger;

  // Create a Network Definition
  IBuilder* builder = createInferBuilder(logger);
  INetworkDefinition* network = builder->createNetworkV2(0);
  
  // Add the Input layer to the network
  int batch_size = 3;
  Dims3 input_shape{4, 2, 2};
  auto input_data = network->addInput("input", DataType::kFLOAT, input_shape);
  
  // Add the plugin layer with hidden layer
  IPluginV2* pluginObj = new LReLU(0.1);
  std::cout << "Add Plugin" << std::endl;
  auto layer = network->addPluginV2(&input_data, 1, *pluginObj);

  // Add a name for the output of the layer so that the tensor can be bound to a memory buffer at inference time:
  layer->getOutput(0)->setName("output");
  // Mark it as the output of the entire network:
  network->markOutput(*layer->getOutput(0));

  // Building an Engine(optimize the network)
  IBuilderConfig* config = builder->createBuilderConfig();
  builder->setMaxBatchSize(3);

  // 序列化和反序列化
  std::cout << "BuildSerializedNetwork" << std::endl;
  IHostMemory*  serializedModel = builder->buildSerializedNetwork(*network, *config);
  IRuntime* runtime = createInferRuntime(logger);

  std::cout << "DeserializeCudaEngine" << std::endl;
  ICudaEngine* engine = runtime->deserializeCudaEngine(serializedModel->data(), serializedModel->size());
  
  // ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);


  // Prepare input_data
  int32_t inputIndex = engine->getBindingIndex("input");
  int32_t outputIndex = engine->getBindingIndex("output");
  
  std::vector<float> input(batch_size * ProductOfDims(input_shape), -1.0);
  std::vector<float> output(batch_size * ProductOfDims(engine->getBindingDimensions(outputIndex)));

  void *GPU_input_Buffer_ptr;  // a host ptr point to a GPU buffer
  void *GPU_output_Buffer_ptr;  // a host ptr point to a GPU buffer

  void* buffers[2];
  cudaMalloc(&GPU_input_Buffer_ptr, sizeof(float)*input.size()); //malloc gpu buffer for input
  cudaMalloc(&GPU_output_Buffer_ptr, sizeof(float)*output.size()); //malloc gpu buffer for output
  cudaMemcpy(GPU_input_Buffer_ptr, input.data(), input.size()*sizeof(float), cudaMemcpyHostToDevice); // copy input data from cpu to gpu
  
  buffers[inputIndex] = static_cast<void*>(GPU_input_Buffer_ptr);
  buffers[outputIndex] = static_cast<void*>(GPU_output_Buffer_ptr);

  // Performing Inference
  IExecutionContext *context = engine->createExecutionContext();
  std::cout << "Execute" << std::endl;
  context->execute(batch_size, buffers);
  // context->execute(batch_size, buffers); // 多次执行

  // copy result data from gpu to cpu
  cudaMemcpy(output.data(), GPU_output_Buffer_ptr, output.size()*sizeof(float), cudaMemcpyDeviceToHost); 

  // display output
  std::cout << "output shape : " << DimsToStr(engine->getBindingDimensions(outputIndex)) << "\n";
  std::cout << "output data : \n";
  for(auto i : output)
    std::cout << i << " ";
  std::cout << std::endl;
}
