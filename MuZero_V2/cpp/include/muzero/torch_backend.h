#pragma once

#include "batcher.h"
#include <torch/script.h>

namespace muzero {

struct TorchLatent : Latent {
    torch::Tensor tensor;
    explicit TorchLatent(torch::Tensor value) : tensor(std::move(value)) {}
};
class TorchBackend : public BatchBackend {
    torch::Device device_;
    torch::jit::Module model_;
    int canvas_size_;
public:
    TorchBackend(const std::string& path, const std::string& device, int canvas_size);
    std::vector<Evaluation> evaluate(const std::vector<std::shared_ptr<InferenceRequest>>& requests) override;
};

}
