#pragma once
#include <vector>
#include <string>
#include <fstream>

namespace model {

class TensorRTInference {
public:
    TensorRTInference(const std::string& engine_file):
                    engine_filename_(engine_file) {}

    virtual ~TensorRTInference();

    bool init();

    void deserilizeEngine();

    void allocateMemory();

    void infer();




private:
    const std::string engine_filename_;
    nvinfer1::ILogger* logger_ = nullptr;
    nvinfer1::IRuntime* runtime_ = nullptr;
    nvinfer1::ICudaEngine* engine_ = nullptr;
    nvinfer1::IExecutionContext* context_ = nullptr;
    std::vector<void*> input_buffers_;
    std::vector<void*> output_buffers_;
};





}