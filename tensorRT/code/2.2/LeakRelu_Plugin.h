
#pragma once

#include "NvInferPlugin.h"
#include <cassert>
#include <iostream>
#include <string>
#include <vector>

using namespace nvinfer1;

class LReLU : public IPluginV2{
public:
    LReLU(float negSlope) : mNegSlope(negSlope) { }

    LReLU(float negSlope, Dims input_dims):mNegSlope(negSlope), input_dims_(input_dims) { }

    LReLU(const void* buffer, size_t length);

    ~LReLU() override = default;

    const char* getPluginType() const noexcept override { return "leak_relu"; }

    const char* getPluginVersion() const noexcept override { return "1"; }

    int getNbOutputs() const noexcept override { return 1; }

    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) noexcept override;
 
    bool supportsFormat(DataType type, PluginFormat format) const noexcept override{ return true; }
    
    void configureWithFormat(Dims const* inputDims, int32_t nbInputs, Dims const* outputDims, int32_t nbOutputs,
        DataType type, PluginFormat format, int32_t maxBatchSize) noexcept override;

    int initialize() noexcept override { 
        return 0; 
    }

    void terminate() noexcept override { }

    size_t getWorkspaceSize(int maxBatchSize) const noexcept override { return 0; }

    int enqueue(int batchSize, const void* const* inputs, void* const* outputs, void* workspace,
        cudaStream_t stream) noexcept override;

    size_t getSerializationSize() const noexcept override { 
        return sizeof(float) + sizeof(Dims); // mNegSlope + input_dims
    }

    void serialize(void* buffer) const noexcept override;

    void destroy() noexcept override { 
        delete this; 
    }

    IPluginV2* clone() const noexcept override;
    
    void setPluginNamespace(const char* libNamespace) noexcept override { namespace_ = libNamespace; }

    const char* getPluginNamespace() const noexcept override { return namespace_.c_str(); }

private:
    float mNegSlope;
    Dims input_dims_;
    std::string namespace_;
};

class LReluPluginCreator : public IPluginCreator{
public:
    LReluPluginCreator() { }

    ~LReluPluginCreator() override = default;

    const char* getPluginName() const noexcept override { return "leak_relu"; }

    const char* getPluginVersion() const noexcept override { return "1"; }

    void setPluginNamespace(AsciiChar const* pluginNamespace) noexcept override { 
        plugin_namespace_ = pluginNamespace;
    }

    const char* getPluginNamespace() const noexcept override { 
        return plugin_namespace_.c_str();
    }

    const PluginFieldCollection* getFieldNames() noexcept override { return nullptr; }

    IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override {
        const PluginField* fields = fc->fields;
        assert(fc->nbFields == 1);
        float negSlope = *(static_cast<const float*>(fields[0].data));
        return new LReLU(negSlope);
    }

    IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override { 
        return new LReLU(serialData, serialLength); 
    }

private:
    std::string plugin_namespace_;
};

REGISTER_TENSORRT_PLUGIN(LReluPluginCreator);