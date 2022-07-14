#pragma once

#include <sstream>
#include "NvInfer.h"

using namespace nvinfer1;

inline size_t ProductOfDims(Dims dims) {
  size_t result = 1;
  for(size_t i = 0; i < dims.nbDims; i++) {
    result *= dims.d[i];
  }
  return result;
}

inline std::string DimsToStr(Dims dims) {
  std::stringstream ss;
  for(size_t i = 0; i < dims.nbDims; i++) {
    ss << dims.d[i] << " ";
  }
  return ss.str();
}

// Write values into buffer
template <typename T>
void write(char*& buffer, const T& val)
{
    memcpy(buffer, &val, sizeof(T));
    buffer += sizeof(T);
}

// Read values from buffer
template <typename T>
T read(const char*& buffer)
{
    T val{};
    memcpy(&val, buffer, sizeof(T));
    buffer += sizeof(T);
    return val;
}