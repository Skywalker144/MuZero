#include "muzero/record.h"
#include "muzero/torch_backend.h"
#include <atomic>
#include <csignal>
#include <iostream>
#include <mutex>
#include <thread>
#include <ATen/Parallel.h>

namespace {
static_assert(std::atomic<bool>::is_always_lock_free);
std::atomic<bool> interrupted{false};
void stop(int) { interrupted.store(true, std::memory_order_relaxed); }
}

int main(int argc, char** argv) {
    using namespace muzero;
    try {
        if (argc != 7) throw std::runtime_error("Usage: muzero_selfplay resolved.cfg model.pt output_dir total_games seed game_offset");
        Config config(argv[1]);
        SearchConfig search_config(config);
        int size = config.integer("BOARD_SIZE");
        int games = std::stoi(argv[4]), offset = std::stoi(argv[6]), threads = config.integer("NUM_GAME_THREADS");
        if (games < 0 || offset < 0 || threads < 1 || config.integer("TORCH_THREADS") < 1)
            throw std::runtime_error("Invalid runtime counts");
        auto seed = std::stoull(argv[5]);
        std::filesystem::path directory(argv[3]);
        std::filesystem::create_directories(directory);
        std::signal(SIGINT, stop);
        std::signal(SIGTERM, stop);
        at::set_num_threads(config.integer("TORCH_THREADS"));
        BatchEvaluator evaluator(std::make_unique<TorchBackend>(argv[2], config.text("DEVICE"), size),
                                 config.integer("NN_MAX_BATCH_SIZE"), config.integer("NN_BATCH_WAIT_US"));
        std::atomic<int> next{0}, completed{0};
        std::atomic<bool> failed{false};
        std::mutex output_mutex;
        std::exception_ptr failure;
        std::vector<std::thread> workers;
        auto work = [&] {
            try {
                while (!interrupted && !failed) {
                    int index = next++;
                    if (index >= games) break;
                    auto destination = game_path(directory, index + offset);
                    if (std::filesystem::exists(destination)) { ++completed; continue; }
                    std::mt19937_64 random(seed + static_cast<uint64_t>(index + offset) * 0x9e3779b97f4a7c15ULL);
                    Game game(size);
                    Search search(search_config, evaluator, random);
                    std::vector<Step> steps;
                    while (!game.finished() && !interrupted && !failed) {
                        bool cheap = std::bernoulli_distribution(search_config.cheap_probability)(random);
                        Budget budget{cheap ? std::min(search_config.full_search_visits, search_config.cheap_visits) : search_config.full_search_visits, cheap};
                        auto result = search.run(game, budget, true);
                        double temperature = search_config.move_temperature(game.turn(), size);
                        int action = choose_action(result.visit_policy, temperature, random);
                        steps.push_back({game.player(), action, budget, game.observation(), std::move(result.policy_target)});
                        game.play(action);
                    }
                    if (!game.finished()) break;
                    auto temporary = destination;
                    temporary += ".tmp";
                    RecordWriter writer(temporary);
                    writer.game(size, game.winner(), steps);
                    std::filesystem::rename(temporary, destination);
                    int done = ++completed;
                    std::lock_guard<std::mutex> lock(output_mutex);
                    std::cout << "game=" << index + offset << " completed=" << done << '/' << games
                              << " rows=" << steps.size() << " winner=" << game.winner() << std::endl;
                }
            } catch (...) {
                failed = true;
                std::lock_guard<std::mutex> lock(output_mutex);
                if (!failure) failure = std::current_exception();
            }
        };
        try {
            for (int i = 0; i < std::min(threads, games); ++i) workers.emplace_back(work);
        } catch (...) {
            failed = true;
            for (auto& worker : workers) worker.join();
            throw;
        }
        for (auto& worker : workers) worker.join();
        if (failure) std::rethrow_exception(failure);
        std::cout << "nn_requests=" << evaluator.requests << " nn_batches=" << evaluator.batches << std::endl;
        return interrupted ? 130 : 0;
    } catch (const std::exception& error) {
        std::cerr << "selfplay: " << error.what() << std::endl;
        return 1;
    }
}
