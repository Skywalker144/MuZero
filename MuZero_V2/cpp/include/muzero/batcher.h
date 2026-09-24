#pragma once

#include "search.h"
#include <atomic>
#include <condition_variable>
#include <deque>
#include <future>
#include <mutex>
#include <thread>

namespace muzero {

struct InferenceRequest {
    std::vector<float> observation;
    std::shared_ptr<const Latent> hidden;
    int action = -1;
    std::promise<Evaluation> promise;
    bool initial() const { return action < 0; }
};
class BatchBackend {
public:
    virtual ~BatchBackend() = default;
    virtual std::vector<Evaluation> evaluate(const std::vector<std::shared_ptr<InferenceRequest>>& requests) = 0;
};
class BatchEvaluator : public Evaluator {
    std::unique_ptr<BatchBackend> backend_;
    size_t max_batch_;
    int wait_us_;
    std::mutex mutex_;
    std::condition_variable ready_;
    std::deque<std::shared_ptr<InferenceRequest>> queue_;
    bool stopping_ = false;
    std::exception_ptr failure_;
    std::thread server_;
    Evaluation submit(std::shared_ptr<InferenceRequest> request);
    void serve();
public:
    std::atomic<uint64_t> requests{0}, batches{0};
    BatchEvaluator(std::unique_ptr<BatchBackend> backend, int max_batch, int wait_us);
    ~BatchEvaluator() override;
    Evaluation initial(const std::vector<float>& observation) override;
    Evaluation recurrent(const std::shared_ptr<const Latent>& hidden, int action) override;
};

}
