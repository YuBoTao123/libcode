#include "tensorrt_interface.h"

using namespace model;

TensorRTInference::~TensorRTInference() {
    context_->destory();
    engine_->destory();
    runtime_->destory();
}

void TensorRTInference::deserilizeEngine() {
    std::ifstream ifs(engine_filename_, std::ios::binary);
    if (!ifs) return;
    ifs.seekg(0, ifs.end);
    int engine_size = ifs.tellg();
    ifs.seekg(0, ifs.beg);



}