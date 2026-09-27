#include "muzero/torch_backend.h"
#include <c10/core/InferenceMode.h>
#include <cstring>

namespace muzero {

TorchBackend::TorchBackend(const std::string& path, const std::string& device, int canvas_size, double policy_temperature)
    : device_(device), model_(torch::jit::load(path, device_)), canvas_size_(canvas_size), policy_temperature_(policy_temperature) {
    if (!std::isfinite(policy_temperature_) || policy_temperature_ <= 0)
        throw std::runtime_error("Invalid NN policy temperature");
    if (device_.is_cuda()) {
        static std::once_flag fusion_strategy;
        std::call_once(fusion_strategy, [] {
            torch::jit::FusionStrategy strategy{{torch::jit::FusionBehavior::DYNAMIC, 2}};
            torch::jit::setFusionStrategy(strategy);
        });
    }
    model_.eval();
    auto metadata = model_.get_method("metadata")({}).toTuple();
    auto model_canvas = metadata->elements()[0].toInt();
    if (model_canvas < 5 || model_canvas > 25 || (canvas_size && model_canvas != canvas_size) ||
        metadata->elements()[1].toInt() != PROTOCOL_VERSION)
        throw std::runtime_error("Model protocol mismatch");
    canvas_size_ = static_cast<int>(model_canvas);
}

std::vector<Evaluation> TorchBackend::evaluate(const std::vector<std::shared_ptr<InferenceRequest>>& requests) {
    c10::InferenceMode guard;
    std::vector<Evaluation> result(requests.size());
    auto capacity = static_cast<int64_t>(requests.size());
    auto options = torch::TensorOptions().device(torch::kCPU).pinned_memory(device_.is_cuda());
    for (bool initial : {true, false}) {
        std::vector<size_t> indices;
        std::vector<torch::Tensor> inputs;
        if (initial && (!observations_.defined() || observations_.size(0) < capacity))
            observations_ = torch::empty({capacity, INPUT_PLANES, canvas_size_, canvas_size_}, options.dtype(torch::kFloat32));
        if (!initial && (!actions_.defined() || actions_.size(0) < capacity))
            actions_ = torch::empty({capacity}, options.dtype(torch::kInt64));
        for (size_t i = 0; i < requests.size(); ++i) {
            const auto& request = requests[i];
            if (request->initial() != initial) continue;
            indices.push_back(i);
            if (initial) {
                if (request->observation.size() != static_cast<size_t>(INPUT_PLANES * canvas_size_ * canvas_size_))
                    throw std::runtime_error("Observation shape mismatch");
                std::memcpy(observations_.data_ptr<float>() + (indices.size() - 1) * request->observation.size(),
                            request->observation.data(), request->observation.size() * sizeof(float));
            } else {
                auto hidden = std::dynamic_pointer_cast<const TorchLatent>(request->hidden);
                if (!hidden) throw std::runtime_error("Invalid latent state backend");
                inputs.push_back(hidden->tensor);
                actions_.data_ptr<int64_t>()[indices.size() - 1] = request->action;
            }
        }
        if (indices.empty()) continue;
        int64_t count = static_cast<int64_t>(indices.size()), action_count = canvas_size_ * canvas_size_;
        auto input = initial ? observations_.narrow(0, 0, count).to(torch::TensorOptions().device(device_), true)
                             : torch::cat(inputs, 0);
        std::vector<torch::jit::IValue> arguments{input};
        if (!initial)
            arguments.push_back(actions_.narrow(0, 0, count).to(torch::TensorOptions().device(device_), true));
        auto tuple = model_.get_method(initial ? "initial" : "recurrent")(arguments).toTuple();
        const auto& values = tuple->elements();
        if (values.size() != 3) throw std::runtime_error("Expected hidden, policy, WDL");
        auto hidden = values[0].toTensor();
        auto policies = values[1].toTensor();
        auto wdl = values[2].toTensor();
        if (hidden.dim() != 4 || hidden.size(0) != count || policies.dim() != 2 ||
            policies.size(0) != count || policies.size(1) != action_count ||
            wdl.dim() != 2 || wdl.size(0) != count || wdl.size(1) != 3)
            throw std::runtime_error("Model output shape mismatch");
        for (size_t i = 0; i < indices.size(); ++i)
            result[indices[i]].hidden = std::make_shared<TorchLatent>(hidden.slice(0, i, i + 1).clone());
        auto outputs = torch::cat({policies / policy_temperature_, wdl}, 1)
                           .to(torch::kCPU).to(torch::kFloat64).contiguous();
        for (size_t i = 0; i < indices.size(); ++i) {
            const double* row = outputs.data_ptr<double>() + i * (action_count + 3);
            result[indices[i]].logits.assign(row, row + action_count);
            std::copy(row + action_count, row + action_count + 3, result[indices[i]].wdl.begin());
        }
    }
    return result;
}

}
