#include "muzero/torch_backend.h"
#include <ATen/Parallel.h>
#include <c10/core/InferenceMode.h>
#include <iostream>

using namespace muzero;

int main(int argc, char** argv) {
    try {
        if (argc != 4) throw std::runtime_error("Usage: torch_backend_test model.pt device canvas");
        c10::InferenceMode guard;
        at::set_num_threads(1);
        at::globalContext().setAllowTF32CuDNN(false);
        at::globalContext().setAllowTF32CuBLAS(false);
        int canvas = std::stoi(argv[3]);
        torch::Device device(argv[2]);
        auto reference = torch::jit::load(argv[1], device);
        reference.eval();
        TorchBackend backend(argv[1], argv[2], canvas, 1.1);
        Game game(canvas, canvas, Rule::FREESTYLE);
        auto observation = game.observation();
        auto input = torch::from_blob(observation.data(), {1, INPUT_PLANES, canvas, canvas}, torch::kFloat32).to(device);
        auto root = reference.get_method("initial")({input}).toTuple()->elements();
        auto retained = std::make_shared<TorchLatent>(root[0].toTensor().clone());
        auto saved = retained->tensor.clone();
        for (int count : {1, 8, 3, 16, 2}) {
            std::vector<std::shared_ptr<InferenceRequest>> requests;
            for (int i = 0; i < count; ++i) {
                auto request = std::make_shared<InferenceRequest>();
                if (i % 2) {
                    request->hidden = retained;
                    request->action = i % (canvas * canvas);
                } else request->observation = observation;
                requests.push_back(std::move(request));
            }
            auto result = backend.evaluate(requests);
            if (result.size() != requests.size()) throw std::runtime_error("Result count mismatch");
            for (int i = 0; i < count; ++i) {
                auto expected = requests[i]->initial() ? root : reference.get_method("recurrent")({
                    saved, torch::tensor({requests[i]->action}, torch::TensorOptions().dtype(torch::kInt64).device(device))
                }).toTuple()->elements();
                auto hidden = std::dynamic_pointer_cast<const TorchLatent>(result[i].hidden);
                if (!hidden) throw std::runtime_error("Missing Torch latent");
                auto logits = torch::from_blob(result[i].logits.data(), {1, canvas * canvas}, torch::kFloat64);
                if (!torch::allclose(hidden->tensor, expected[0].toTensor(), 1e-4, 1e-5) ||
                    !torch::allclose(logits, (expected[1].toTensor() / 1.1).to(torch::kCPU).to(torch::kFloat64), 1e-4, 1e-5) ||
                    !torch::allclose(torch::from_blob(result[i].wdl.data(), {1, 3}, torch::kFloat64),
                                     expected[2].toTensor().to(torch::kCPU).to(torch::kFloat64), 1e-4, 1e-5))
                    throw std::runtime_error("Backend differs from TorchScript reference at batch " + std::to_string(count) +
                        " row " + std::to_string(i) + " hidden error " +
                        std::to_string((hidden->tensor - expected[0].toTensor()).abs().max().item<double>()));
                if (hidden->tensor.storage().nbytes() != hidden->tensor.nbytes())
                    throw std::runtime_error("Latent retains an entire batch allocation");
            }
            if (!torch::equal(saved, retained->tensor)) throw std::runtime_error("Retained latent was overwritten");
            retained = std::make_shared<TorchLatent>(
                std::dynamic_pointer_cast<const TorchLatent>(result.back().hidden)->tensor);
            saved = retained->tensor.clone();
        }
        std::cout << "Backend reference and latent lifetime checks passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
