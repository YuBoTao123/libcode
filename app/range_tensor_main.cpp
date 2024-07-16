#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "cuda_runtime_api.h"
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>
using namespace nvinfer1;
using namespace std;

class Logger : public ILogger {
  void log(Severity severity, const char *msg) override {
    // Log messages
  }
} gLogger;

// Helper function to calculate volume of a shape
int64_t volume(const Dims &dims) {
  int64_t v = 1;
  for (int i = 0; i < dims.nbDims; ++i) { // 维度的数量
    v *= dims.d[i];                       // 每一个维度的维度
  }
  return v;
}

int main() {
  // Step 1: Load TensorRT engine
  std::string engineFile = "salsanext.engine";
  std::ifstream engineStream(engineFile, std::ios::binary);
  if (!engineStream) {
    std::cerr << "Error opening engine file." << std::endl;
    return 1;
  }

  engineStream.seekg(0, engineStream.end);
  int engineSize = engineStream.tellg();
  engineStream.seekg(0, engineStream.beg);
  std::unique_ptr<char[]> engineData(new char[engineSize]);
  engineStream.read(engineData.get(), engineSize);
  engineStream.close();

  IRuntime *runtime = createInferRuntime(gLogger);
  ICudaEngine *engine =
      runtime->deserializeCudaEngine(engineData.get(), engineSize, nullptr);
  if (!engine) {
    std::cerr << "Error deserializing engine." << std::endl;
    return 1;
  }

  // Step 2: Create execution context
  IExecutionContext *context = engine->createExecutionContext();
  int inputIndex =
      engine->getBindingIndex("input"); // Replace with actual input name
  int outputIndex =
      engine->getBindingIndex("output"); // Replace with actual output name

  // if (inputIndex < 0 || outputIndex < 0) {
  //     std::cerr << "Input or output binding not found." << std::endl;
  //     return 1;
  // }

  // Step 3: Allocate input and output memory
  const int maxBatchSize = engine->getMaxBatchSize();
  // Dims inputDims = engine->getBindingDimensions(inputIndex);
  // Dims outputDims = engine->getBindingDimensions(outputIndex);
  Dims inputDims{4, {1, 6, 64, 2048}};
  Dims outputDims{4, {1, 3, 64, 2048}};

  int inputSize = volume(inputDims) * maxBatchSize;
  int outputSize = volume(outputDims) * maxBatchSize;

  float *inputHostMem = new float[inputSize];
  float *outputHostMem = new float[outputSize];

  void *inputDeviceMem;
  void *outputDeviceMem;
  cudaMalloc(&inputDeviceMem, inputSize * sizeof(float));
  cudaMalloc(&outputDeviceMem, outputSize * sizeof(float));

  // Step 4: Prepare input data
  float *inputData = new float[inputSize];
  for (int i = 0; i < inputSize; ++i) {
    inputData[i] = static_cast<float>(rand()) /
                   RAND_MAX; // Replace with your actual input data
  }

  // Step 5: Execute inference
  void *bindings[] = {inputDeviceMem, outputDeviceMem};
  context->execute(maxBatchSize, bindings);

  // Step 6: Retrieve output data
  cudaMemcpy(outputHostMem, outputDeviceMem, outputSize * sizeof(float),
             cudaMemcpyDeviceToHost);

  // Step 7: Process and output inference results
  // Replace with your actual post-processing code
  std::cout << "Inference result:" << std::endl;
  for (int i = 0; i < outputSize; ++i) {
    std::cout << outputHostMem[i] << " ";
    if ((i + 1) % outputDims.d[0] == 0) {
      std::cout << std::endl;
    }
  }

  // Step 8: Clean up
  delete[] inputHostMem;
  delete[] outputHostMem;
  cudaFree(inputDeviceMem);
  cudaFree(outputDeviceMem);
  context->destroy();
  engine->destroy();
  runtime->destroy();

  return 0;
}
