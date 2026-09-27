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
    double policy_temperature_;
    torch::Tensor observations_, actions_;
public:
    TorchBackend(const std::string& path, const std::string& device, int canvas_size = 0, double policy_temperature = 1.0);
    int canvas_size() const { return canvas_size_; }
    std::vector<Evaluation> evaluate(const std::vector<std::shared_ptr<InferenceRequest>>& requests) override;
};

}
