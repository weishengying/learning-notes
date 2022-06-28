// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include "NvInferRuntime.h"
#include "NvInferRuntimeCommon.h"
#include <stdio.h>

#include <cassert>
#include <string>
#include <vector>

#include "plugin_utils.h"

class SwishPlugin : public nvinfer1::IPluginV2IOExt {
 public:
  explicit SwishPlugin(float beta) : beta_(beta) { }

  SwishPlugin(float beta, std::vector<int>input_shape) : beta_(beta), input_shape_(input_shape) { }

  SwishPlugin(void const* serialData, size_t serialLength);

  // IPluginV2 methods
  const char* getPluginType() const noexcept override;
  const char* getPluginVersion() const noexcept override;
  int getNbOutputs() const noexcept override;
  nvinfer1::Dims getOutputDimensions(int outputIndex,
                                     const nvinfer1::Dims* inputs,
                                     int nbInputs) noexcept override;
  int32_t initialize() noexcept override;
  void terminate() noexcept override;
  size_t getWorkspaceSize(int32_t maxBatchSize) const noexcept override;
  int enqueue(int batchSize,
              const void* const* inputs,
              void* const* outputs,
              void* workspace,
              cudaStream_t stream) noexcept override;
  size_t getSerializationSize() const noexcept override;
  void serialize(void* buffer) const noexcept override;
  void destroy() noexcept override;
  void setPluginNamespace(char const* pluginNamespace) noexcept override;
  const char* getPluginNamespace() const noexcept override;

  // IPluginV2Ext methods
  nvinfer1::DataType getOutputDataType(int32_t index,
                                       nvinfer1::DataType const* inputTypes,
                                       int32_t nbInputs) const
      noexcept override;
  bool isOutputBroadcastAcrossBatch(int32_t outputIndex,
                                    bool const* inputIsBroadcasted,
                                    int32_t nbInputs) const noexcept override;
  bool canBroadcastInputAcrossBatch(int32_t inputIndex) const noexcept override;

  IPluginV2Ext* clone() const noexcept override;

  // IPluginV2IOExt methods
  void configurePlugin(nvinfer1::PluginTensorDesc const* in,
                       int32_t nb_input,
                       nvinfer1::PluginTensorDesc const* out,
                       int32_t nb_output) noexcept override;
  bool supportsFormatCombination(int32_t pos,
                                 nvinfer1::PluginTensorDesc const* inOut,
                                 int32_t nb_inputs,
                                 int32_t nb_outputs) const noexcept override;

 private:
  float beta_;
  std::vector<int> input_shape_;

 private:
  nvinfer1::Dims input_dims_;
  nvinfer1::DataType data_type_;
  nvinfer1::PluginFormat data_format_;
  std::string namespace_;
};

class SwishPluginCreator : public nvinfer1::IPluginCreator {
 public:
  const char* getPluginName() const noexcept override { return "swish_plugin"; }

  const char* getPluginVersion() const noexcept override { return "1"; }

  const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override {
    return &field_collection_;
  }

  nvinfer1::IPluginV2* createPlugin(
      const char* name,
      const nvinfer1::PluginFieldCollection* fc) noexcept override;

  nvinfer1::IPluginV2* deserializePlugin(const char* name,
                                         const void* serialData,
                                         size_t serialLength) noexcept override;

  void setPluginNamespace(const char* plugin_namespace) noexcept override { 
    plugin_namespace_ = plugin_namespace; 
  }

  const char* getPluginNamespace() const noexcept override { 
    return plugin_namespace_.c_str(); 
  }

 private:
  std::string plugin_namespace_;
  std::string plugin_name_;
  nvinfer1::PluginFieldCollection field_collection_{0, nullptr};
};
REGISTER_TENSORRT_PLUGIN(SwishPluginCreator);

