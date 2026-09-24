#include "muzero/torch_backend.h"
#include <c10/core/InferenceMode.h>

namespace muzero {

TorchBackend::TorchBackend(const std::string& path, const std::string& device, int board_size)
    : device_(device), model_(torch::jit::load(path, device_)), board_size_(board_size) {
    model_.eval();
    auto metadata = model_.get_method("metadata")({}).toTuple();
    if (metadata->elements()[0].toInt() != board_size || metadata->elements()[1].toInt() != 2)
        throw std::runtime_error("Model protocol mismatch");
}

std::vector<Evaluation> TorchBackend::evaluate(const std::vector<std::shared_ptr<InferenceRequest>>& requests) {
    c10::InferenceMode guard;
    std::vector<Evaluation> result(requests.size());
    for (bool initial : {true, false}) {
        std::vector<size_t> indices;
        std::vector<torch::Tensor> inputs;
        std::vector<int64_t> actions;
        for (size_t i = 0; i < requests.size(); ++i) {
            const auto& request = requests[i];
            if (request->initial() != initial) continue;
            indices.push_back(i);
            if (initial) {
                if (request->observation.size() != static_cast<size_t>(3 * board_size_ * board_size_))
                    throw std::runtime_error("Observation shape mismatch");
                inputs.push_back(torch::from_blob(request->observation.data(), {1, 3, board_size_, board_size_}, torch::kFloat32));
            } else {
                auto hidden = std::dynamic_pointer_cast<const TorchLatent>(request->hidden);
                if (!hidden) throw std::runtime_error("Invalid latent state backend");
                inputs.push_back(hidden->tensor);
                actions.push_back(request->action);
            }
        }
        if (indices.empty()) continue;
        auto input = torch::cat(inputs, 0).to(device_);
        std::vector<torch::jit::IValue> arguments{input};
        if (!initial)
            arguments.push_back(torch::from_blob(actions.data(), {static_cast<int64_t>(actions.size())}, torch::kInt64).to(device_));
        auto tuple = model_.get_method(initial ? "initial" : "recurrent")(arguments).toTuple();
        const auto& values = tuple->elements();
        if (values.size() != 3) throw std::runtime_error("Expected hidden, policy, value");
        auto hidden = values[0].toTensor();
        auto policies = values[1].toTensor().to(torch::kCPU).to(torch::kFloat64).contiguous();
        auto utilities = values[2].toTensor().to(torch::kCPU).to(torch::kFloat64).contiguous();
        int64_t count = static_cast<int64_t>(indices.size()), action_count = board_size_ * board_size_;
        if (hidden.dim() != 4 || hidden.size(0) != count || policies.dim() != 2 ||
            policies.size(0) != count || policies.size(1) != action_count || utilities.numel() != count)
            throw std::runtime_error("Model output shape mismatch");
        for (size_t i = 0; i < indices.size(); ++i) {
            const double* row = policies.data_ptr<double>() + i * action_count;
            result[indices[i]] = {std::make_shared<TorchLatent>(hidden.slice(0, i, i + 1).clone()),
                                 std::vector<double>(row, row + action_count), utilities.data_ptr<double>()[i]};
        }
    }
    return result;
}

}
