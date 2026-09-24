#include "muzero/batcher.h"
#include <chrono>

namespace muzero {

BatchEvaluator::BatchEvaluator(std::unique_ptr<BatchBackend> backend, int max_batch, int wait_us)
    : backend_(std::move(backend)), max_batch_(max_batch), wait_us_(wait_us) {
    if (!backend_ || max_batch < 1 || wait_us < 0) throw std::runtime_error("Invalid batch configuration");
    server_ = std::thread(&BatchEvaluator::serve, this);
}

BatchEvaluator::~BatchEvaluator() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
    }
    ready_.notify_all();
    server_.join();
}

Evaluation BatchEvaluator::submit(std::shared_ptr<InferenceRequest> request) {
    auto future = request->promise.get_future();
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (failure_) std::rethrow_exception(failure_);
        if (stopping_) throw std::runtime_error("Evaluator is stopping");
        queue_.push_back(std::move(request));
    }
    ready_.notify_one();
    return future.get();
}

Evaluation BatchEvaluator::initial(const std::vector<float>& observation) {
    auto request = std::make_shared<InferenceRequest>();
    request->observation = observation;
    return submit(std::move(request));
}

Evaluation BatchEvaluator::recurrent(const std::shared_ptr<const Latent>& hidden, int action) {
    if (action < 0 || !hidden) throw std::runtime_error("Invalid recurrent inference request");
    auto request = std::make_shared<InferenceRequest>();
    request->hidden = hidden;
    request->action = action;
    return submit(std::move(request));
}

void BatchEvaluator::serve() {
    for (;;) {
        std::vector<std::shared_ptr<InferenceRequest>> batch;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            ready_.wait(lock, [&] { return stopping_ || !queue_.empty(); });
            if (queue_.empty() && stopping_) return;
            auto deadline = std::chrono::steady_clock::now() + std::chrono::microseconds(wait_us_);
            ready_.wait_until(lock, deadline, [&] { return stopping_ || queue_.size() >= max_batch_; });
            while (!queue_.empty() && batch.size() < max_batch_) {
                batch.push_back(std::move(queue_.front()));
                queue_.pop_front();
            }
        }
        try {
            auto outputs = backend_->evaluate(batch);
            if (outputs.size() != batch.size()) throw std::runtime_error("Inference batch size mismatch");
            bool initial = false, recurrent = false;
            for (const auto& request : batch) {
                initial |= request->initial();
                recurrent |= !request->initial();
            }
            requests += batch.size();
            batches += static_cast<int>(initial) + static_cast<int>(recurrent);
            for (size_t i = 0; i < batch.size(); ++i) batch[i]->promise.set_value(std::move(outputs[i]));
        } catch (...) {
            auto failure = std::current_exception();
            std::lock_guard<std::mutex> lock(mutex_);
            failure_ = failure;
            for (auto& request : batch) request->promise.set_exception(failure);
            for (auto& request : queue_) request->promise.set_exception(failure);
            queue_.clear();
            return;
        }
    }
}

}
