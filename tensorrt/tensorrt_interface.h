#pragma once
#include "NvInfer.h"
#include <fstream>
#include <string>
#include <vector>
#include <iostream>
#include <numeric>

namespace model {

using namespace nvinfer1;

class Logger : public ILogger {
  void log(Severity severity, const char *msg) noexcept override {
    // suppress info-level messages
      if (severity <= Severity::kWARNING)
          std::cout << msg << std::endl;
  }
};

class TensorRTInference {
public:
  TensorRTInference(const std::string &engine_file);

  virtual ~TensorRTInference();

  bool init();

  void deserilizeEngine();

  bool allocateMemory();

  void infer(float *in_data, float *out_data);

  int64_t volume(const nvinfer1::Dims &d) {
    return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
  }

private:
  const std::string engine_filename_;
  Logger logger_;

  nvinfer1::IRuntime *runtime_ = nullptr;
  nvinfer1::ICudaEngine *engine_ = nullptr;
  nvinfer1::IExecutionContext *context_ = nullptr;
  std::vector<void *> input_buffers_;
  std::vector<void *> output_buffers_;

  int max_batch_size_ = 1;
  int input_size_ = 0;
  int output_size_ = 0;
  // host memmory
  std::vector<float> input_host_mem_;
  std::vector<float> output_host_mem_;

  // device memory
  void *input_device_mem_ = nullptr;
  void *output_device_mem_ = nullptr;
};

} // namespace model