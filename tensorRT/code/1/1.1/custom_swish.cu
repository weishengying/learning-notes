#include "swish_op.h"
#include "plugin_utils.h"
#include "iostream"
template <typename T>
__device__ T math_exp(T a);


template <>
__device__ float math_exp<float>(float a) {
  return expf(a);
}

template <typename T>
__global__ void swish_kernel(int num, const T *input, T *output, T beta) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < num) {
#if __CUDA_ARCH__ >= 350
    output[index] =
        __ldg(input + index) /
        (static_cast<T>(1.0) + math_exp<T>(-beta * __ldg(input + index)));
#else
    output[index] = input[index] /
                    (static_cast<T>(1.0) + math_exp<T>(-beta * input[index]));
#endif
  }
}

SwishPlugin::SwishPlugin(void const* serialData, size_t serialLength) {
  DeserializeValue<float>(&serialData, &serialLength, &beta_);
  DeserializeValue<std::vector<int>>(&serialData, &serialLength, &input_shape_);
}

const char* SwishPlugin::getPluginType() const noexcept { return "swish_plugin"; }

const char* SwishPlugin::getPluginVersion() const noexcept { return "1"; }

int SwishPlugin::getNbOutputs() const noexcept { return 1; }

nvinfer1::Dims SwishPlugin::getOutputDimensions(int outputIndex,
                                               const nvinfer1::Dims* inputs,
                                               int nbInputs) noexcept {
  assert(nbInputs == 1);
  assert(inputs[0].nbDims == 3);
  return *inputs;
}

int32_t SwishPlugin::initialize() noexcept { return 0; }

void SwishPlugin::terminate() noexcept {}

size_t SwishPlugin::getWorkspaceSize(int32_t maxBatchSize) const noexcept {
  return 0;
}

int SwishPlugin::enqueue(int batch_size,
                        const void* const* inputs,
                        void* const* outputs,
                        void* workspace,
                        cudaStream_t stream) noexcept {
  // input dims is CHW.
  const float *input = reinterpret_cast<const float *>(inputs[0]);
  float *output = reinterpret_cast<float *const *>(outputs)[0];
  int num = batch_size;
  for (int i = 0; i < input_shape_.size(); i++) {
    num *= input_shape_[i];
  }

  int threads = 1024;
  int blocks = (num + threads - 1) / threads;
  swish_kernel<<<blocks, threads, 0, stream>>>(num, input, output, beta_);

  return cudaGetLastError() != cudaSuccess;
}

size_t SwishPlugin::getSerializationSize() const noexcept {
  return SerializedSize(beta_) + SerializedSize(input_shape_);
}

void SwishPlugin::serialize(void* buffer) const noexcept {
  SerializeValue(&buffer, beta_);
  SerializeValue(&buffer, input_shape_);
}

void SwishPlugin::destroy() noexcept { delete this; }

void SwishPlugin::setPluginNamespace(char const* plugin_namespace) noexcept {
  namespace_ = plugin_namespace;
}

const char* SwishPlugin::getPluginNamespace() const noexcept {
  return namespace_.c_str();
}

nvinfer1::DataType SwishPlugin::getOutputDataType(
    int32_t index,
    nvinfer1::DataType const* input_types,
    int32_t nbInputs) const noexcept {
  assert(index == 0);
  assert((input_types[0] == nvinfer1::DataType::kFLOAT) == true);
  return input_types[0];
}

bool SwishPlugin::isOutputBroadcastAcrossBatch(int32_t outputIndex,
                                              bool const* inputIsBroadcasted,
                                              int32_t nbInputs) const noexcept {
  return false;
}

bool SwishPlugin::canBroadcastInputAcrossBatch(int32_t inputIndex) const
    noexcept {
  return false;
}

nvinfer1::IPluginV2Ext* SwishPlugin::clone() const noexcept {
  auto* plugin = new SwishPlugin(beta_, input_shape_);
  plugin->setPluginNamespace(namespace_.c_str());
  return plugin;
}

void SwishPlugin::configurePlugin(nvinfer1::PluginTensorDesc const* in,
                                 int32_t nb_input,
                                 nvinfer1::PluginTensorDesc const* out,
                                 int32_t nb_output) noexcept {
  assert(nb_input == 1);
  assert(nb_output == 1);
  input_dims_ = in[0].dims;
  input_shape_.resize(input_dims_.nbDims);
  for(size_t i = 0; i < input_shape_.size(); i++) input_shape_[i] = input_dims_.d[i];
  data_format_ = in[0].format;
  data_type_ = in[0].type;
}

bool SwishPlugin::supportsFormatCombination(
    int32_t pos,
    nvinfer1::PluginTensorDesc const* in_out,
    int32_t nb_inputs,
    int32_t nb_outputs) const noexcept {
  assert(pos < nb_inputs + nb_outputs);
  assert(in_out);

  return ((in_out[pos].type == nvinfer1::DataType::kFLOAT) &&
          in_out[pos].format == nvinfer1::PluginFormat::kLINEAR);
}

nvinfer1::IPluginV2* SwishPluginCreator::createPlugin(
    const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept {
  return nullptr;
}

nvinfer1::IPluginV2* SwishPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept {
  auto* plugin = new SwishPlugin(serialData, serialLength);
  plugin_name_ = name;
  return plugin;
}
