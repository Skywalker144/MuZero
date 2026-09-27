#include "muzero/torch_backend.h"
#include <ATen/Parallel.h>
#include <chrono>
#include <iostream>
#include <map>

using namespace muzero;
using Clock = std::chrono::steady_clock;

class MeasuredBackend final : public BatchBackend {
    TorchBackend backend_;
public:
    double seconds = 0;
    std::map<size_t, uint64_t> initial_batches, recurrent_batches;
    MeasuredBackend(const std::string& model, const Config& c)
        : backend_(model, c.text("DEVICE"), GameConfig(c).canvas) {}
    std::vector<Evaluation> evaluate(const std::vector<std::shared_ptr<InferenceRequest>>& requests) override {
        auto start = Clock::now();
        auto result = backend_.evaluate(requests);
        seconds += std::chrono::duration<double>(Clock::now() - start).count();
        size_t initial = 0;
        for (const auto& request : requests) initial += request->initial();
        if (initial) ++initial_batches[initial];
        if (initial < requests.size()) ++recurrent_batches[requests.size() - initial];
        return result;
    }
    void reset() {
        seconds = 0;
        initial_batches.clear();
        recurrent_batches.clear();
    }
};

int main(int argc, char** argv) {
    try {
        if (argc != 4) throw std::runtime_error("Usage: muzero_benchmark resolved.cfg model.pt searches_per_thread");
        Config c(argv[1]);
        SearchConfig search_config(c);
        GameConfig game_config(c);
        int threads = c.integer("NUM_GAME_THREADS"), searches = std::stoi(argv[3]);
        if (threads < 1 || searches < 1) throw std::runtime_error("Invalid benchmark counts");
        at::set_num_threads(c.integer("TORCH_THREADS"));
        auto backend = std::make_unique<MeasuredBackend>(argv[2], c);
        auto* measured = backend.get();
        BatchEvaluator evaluator(std::move(backend), c.integer("NN_MAX_BATCH_SIZE"), c.integer("NN_BATCH_WAIT_US"));
        std::mutex mutex;
        std::condition_variable ready;
        int warmed = 0;
        bool start = false;
        std::exception_ptr failure;
        std::vector<std::future<void>> workers;
        try {
            for (int i = 0; i < threads; ++i) workers.push_back(std::async(std::launch::async, [&, i] {
                try {
                    std::mt19937_64 random(static_cast<uint64_t>(c.integer("SEED")) + i);
                    auto [size, rule] = game_config.sample(random);
                    Game game(size, game_config.canvas, rule);
                    Search search(search_config, evaluator, random);
                    search.run(game, {search_config.full_search_visits, false}, false);
                    {
                        std::unique_lock<std::mutex> lock(mutex);
                        ++warmed;
                        ready.notify_all();
                        ready.wait(lock, [&] { return start; });
                    }
                    for (int j = 0; j < searches; ++j)
                        search.run(game, {search_config.full_search_visits, false}, false);
                } catch (...) {
                    std::lock_guard<std::mutex> lock(mutex);
                    if (!failure) failure = std::current_exception();
                    ready.notify_all();
                }
            }));
        } catch (...) {
            std::lock_guard<std::mutex> lock(mutex);
            if (!failure) failure = std::current_exception();
        }
        auto begin = Clock::now();
        uint64_t requests = 0, batches = 0;
        {
            std::unique_lock<std::mutex> lock(mutex);
            ready.wait(lock, [&] { return warmed == threads || failure; });
            if (!failure) {
                measured->reset();
                requests = evaluator.requests.load();
                batches = evaluator.batches.load();
                begin = Clock::now();
            }
            start = true;
            ready.notify_all();
        }
        for (auto& worker : workers) worker.get();
        if (failure) std::rethrow_exception(failure);
        double seconds = std::chrono::duration<double>(Clock::now() - begin).count();
        requests = evaluator.requests.load() - requests;
        batches = evaluator.batches.load() - batches;
        std::cout << "{\"seconds\":" << seconds << ",\"backend_seconds\":" << measured->seconds
                  << ",\"requests\":" << requests << ",\"batches\":" << batches
                  << ",\"requests_per_second\":" << requests / seconds
                  << ",\"mean_batch\":" << static_cast<double>(requests) / batches
                  << ",\"threads\":" << threads << ",\"max_batch\":" << c.integer("NN_MAX_BATCH_SIZE")
                  << ",\"wait_us\":" << c.integer("NN_BATCH_WAIT_US")
                  << ",\"searches_per_thread\":" << searches << ",\"visits\":" << search_config.full_search_visits;
        for (const auto& entry : {std::make_pair("initial_batches", &measured->initial_batches),
                                  std::make_pair("recurrent_batches", &measured->recurrent_batches)}) {
            std::cout << ",\"" << entry.first << "\":{";
            bool first = true;
            for (const auto& [size, count] : *entry.second) {
                if (!first) std::cout << ',';
                first = false;
                std::cout << '\"' << size << "\":" << count;
            }
            std::cout << '}';
        }
        std::cout << "}\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
