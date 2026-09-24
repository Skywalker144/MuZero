#include "muzero/batcher.h"
#include <iostream>

using namespace muzero;

class Backend : public BatchBackend {
    bool fail_;
public:
    explicit Backend(bool fail) : fail_(fail) {}
    std::vector<Evaluation> evaluate(const std::vector<std::shared_ptr<InferenceRequest>>& requests) override {
        if (fail_) throw std::runtime_error("injected backend failure");
        std::vector<Evaluation> result;
        for (const auto& request : requests)
            result.push_back({nullptr, {static_cast<double>(request->observation[0])}, 0.5});
        return result;
    }
};

int main() {
    try {
        for (bool fail : {false, true}) {
            BatchEvaluator evaluator(std::make_unique<Backend>(fail), 8, 1000);
            std::vector<std::future<bool>> clients;
            for (int i = 0; i < 24; ++i) clients.push_back(std::async(std::launch::async, [&, i] {
                try {
                    auto output = evaluator.initial({static_cast<float>(i)});
                    return !fail && output.logits[0] == i;
                } catch (const std::runtime_error&) { return fail; }
            }));
            for (auto& client : clients)
                if (!client.get()) throw std::runtime_error("Batch result or failure propagation mismatch");
        }
        std::cout << "batcher invariants passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
