
#pragma once

#include "NvInferPlugin.h"
#include <cassert>
#include <iostream>
#include <string>
#include <vector>

using namespace nvinfer1;

class LReLU : public IPluginV2DynamicExt{
public:
    LReLU(float negSlope) : mNegSlope(negSlope) { }

    LReLU(float negSlope, Dims input_dims):mNegSlope(negSlope), input_dims_(input_dims) { }

    LReLU(const void* buffer, size_t length);

    ~LReLU() override = default;

    const char* getPluginType() const noexcept override { return "leak_relu"; }

    const char* getPluginVersion() const noexcept override { return "1"; }

    int getNbOutputs() const noexcept override { 
        return 1; 
    }

    int initialize() noexcept override { 
        return 0; 
    }

    void terminate() noexcept override { }

    size_t getSerializationSize() const noexcept override { 
        return sizeof(float) + sizeof(Dims); // mNegSlope + input_dims
    }

    void serialize(void* buffer) const noexcept override;

    void destroy() noexcept override { 
        delete this; 
    }
    
    void setPluginNamespace(const char* libNamespace) noexcept override { namespace_ = libNamespace; }

    const char* getPluginNamespace() const noexcept override { return namespace_.c_str(); }


    /*IPluginV2Ext method*/
    nvinfer1::DataType getOutputDataType(
        int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept override {return inputTypes[index];}

    /*IPluginV2DynamicExt method*/
    IPluginV2DynamicExt* clone() const noexcept override;

    DimsExprs getOutputDimensions(
        int32_t outputIndex, const DimsExprs* inputs, int32_t nbInputs, IExprBuilder& exprBuilder) noexcept;

    bool supportsFormatCombination(
        int32_t pos, const PluginTensorDesc* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept {
            return false;
    }

    void configurePlugin(const DynamicPluginTensorDesc* in, int32_t nbInputs,
        const DynamicPluginTensorDesc* out, int32_t nbOutputs) noexcept {

    }

    size_t getWorkspaceSize(const PluginTensorDesc* inputs, int32_t nbInputs, const PluginTensorDesc* outputs,
        int32_t nbOutputs) const noexcept { return 0; }

    virtual int32_t enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc,
        const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept;
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
        return nullptr;
    }

    IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override { 
        return new LReLU(serialData, serialLength); 
    }

private:
    std::string plugin_namespace_;
};

REGISTER_TENSORRT_PLUGIN(LReluPluginCreator);