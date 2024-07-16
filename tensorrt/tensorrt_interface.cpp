#include "tensorrt_interface.h"
#include <fstream>
#include <iostream>

using namespace model;

TensorRTInference::TensorRTInference(const std::string &engine_file)
    : engine_filename_(engine_file), logger_(), runtime_(nullptr),
      engine_(nullptr), context_(nullptr), input_device_mem_(nullptr),
      output_device_mem_(nullptr) {
  if (!init()) {
    std::cerr << "init TensorRTInference fail" << std::endl;
  }
}

TensorRTInference::~TensorRTInference() {
  if (context_)
    context_->destroy();
  if (engine_)
    engine_->destroy();
  if (runtime_)
    runtime_->destroy();
  cudaFree(input_device_mem_);
  cudaFree(output_device_mem_);
  input_device_mem_ = nullptr;
  output_device_mem_ = nullptr;
}

void TensorRTInference::infer(float *in_data, float *out_data) {
  if (!context_)
    return;
  // host to device
  if (cudaMemcpy(input_device_mem_, in_data, input_size_ * sizeof(float),
                 cudaMemcpyHostToDevice) != cudaSuccess) {
    std::cerr << "cudaMemcpy from host to device fail" << std::endl;
    return;
  }
  void *bindings[] = {input_device_mem_, output_device_mem_};
  context_->executeV2(bindings); // 使用 executeV2 提升性能
  // device to host
  if (cudaMemcpy(out_data, output_device_mem_, output_size_ * sizeof(float),
                 cudaMemcpyDeviceToHost) != cudaSuccess) {
    std::cerr << "cudaMemcpy from device to host fail" << std::endl;
  }
}

bool TensorRTInference::init() {
  deserilizeEngine();
  return allocateMemory();
}

void TensorRTInference::deserilizeEngine() {
  std::ifstream ifs(engine_filename_, std::ios::binary);
  if (!ifs.is_open()) {
    std::cerr << "open engine file fail" << std::endl;
    return;
  }
  ifs.seekg(0, ifs.end);
  int engine_size = ifs.tellg();
  ifs.seekg(0, ifs.beg);
  std::unique_ptr<char[]> engine_data(new char[engine_size]);
  ifs.read(engine_data.get(), engine_size);
  ifs.close();
  runtime_ = nvinfer1::createInferRuntime(logger_);
  if (!runtime_) {
    std::cerr << "create InferRuntime fail" << std::endl;
    return;
  }
  engine_ =
      runtime_->deserializeCudaEngine(engine_data.get(), engine_size, nullptr);
  if (!engine_) {
    runtime_->destroy();
    runtime_ = nullptr;
    std::cerr << "deserialize engine fail" << std::endl;
  }
}

bool TensorRTInference::allocateMemory() {
  if (!engine_)
    return false;
  context_ = engine_->createExecutionContext();
  if (!context_)
    return false;
  // allocate memory
  int inputIndex = engine_->getBindingIndex("input");
  int outputIndex = engine_->getBindingIndex("output");
  Dims input_dims = context_->getBindingDimensions(inputIndex);
  Dims output_dims = context_->getBindingDimensions(outputIndex);
  max_batch_size_ = engine_->getMaxBatchSize();
  input_size_ = volume(input_dims) * max_batch_size_;
  output_size_ = volume(output_dims) * max_batch_size_;
  input_host_mem_.resize(input_size_);
  output_host_mem_.resize(output_size_);

  if (cudaMalloc(&input_device_mem_, input_size_ * sizeof(float)) !=
      cudaSuccess) {
    std::cerr << "allocate fail: input_device_mem_" << std::endl;
    return false;
  }

  if (cudaMalloc(&output_device_mem_, output_size_ * sizeof(float)) !=
      cudaSuccess) {
    std::cerr << "allocate fail: output_device_mem_" << std::endl;
    cudaFree(input_device_mem_);
    input_device_mem_ = nullptr;
    return false;
  }
  return true;
}
