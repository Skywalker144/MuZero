#include "muzero/batcher.h"
#include <chrono>
#include <iostream>
#include <set>

using namespace muzero;

void require(bool condition) {
    if (!condition) throw std::runtime_error("parallel search assertion failed");
}

struct TestLatent : Latent {};

class Backend : public BatchBackend {
public:
    int initial_calls = 0, recurrent_calls = 0, largest_batch = 0;
    bool fail = false;
    std::atomic<bool>* cancellation = nullptr;
    std::set<std::pair<const Latent*, int>> expanded;
    std::vector<std::shared_ptr<const Latent>> retained;
    std::vector<Evaluation> evaluate(const std::vector<std::shared_ptr<InferenceRequest>>& requests) override {
        largest_batch = std::max(largest_batch, static_cast<int>(requests.size()));
        std::vector<Evaluation> outputs;
        for (const auto& request : requests) {
            if (request->initial()) ++initial_calls;
            else {
                ++recurrent_calls;
                require(expanded.emplace(request->hidden.get(), request->action).second);
                if (fail) throw std::runtime_error("injected search failure");
                if (cancellation && recurrent_calls >= 5) cancellation->store(true);
            }
            auto latent = std::make_shared<TestLatent>();
            retained.push_back(latent);
            outputs.push_back({latent, std::vector<double>(25, 0), 0.25});
        }
        return outputs;
    }
};

int main(int argc, char** argv) {
    try {
        require(argc == 2);
        Config raw(argv[1]);
        SearchConfig config(raw);
        Game game(5);
        game.play(12);
        for (int threads : {1, 4, 8}) {
            for (int visits : {1, 7, 64}) {
                auto backend = std::make_unique<Backend>();
                auto* measured = backend.get();
                BatchEvaluator evaluator(std::move(backend), 8, 10000);
                std::mt19937_64 random(3);
                Search search(config, evaluator, random);
                auto result = search.run(game, {visits, false}, false, {threads, 1.0});
                require(result.completed_visits == visits);
                require(measured->initial_calls == 1 && measured->recurrent_calls == visits);
                require(std::accumulate(result.visits.begin(), result.visits.end(), 0) == visits);
                require(result.selection_policy[12] == 0);
                require(std::abs(std::accumulate(result.selection_policy.begin(), result.selection_policy.end(), 0.0) - 1) < 1e-9);
                if (threads > 1 && visits >= 7) require(measured->largest_batch > 1);
            }
        }
        for (bool fail : {false, true}) {
            auto backend = std::make_unique<Backend>();
            std::atomic<bool> cancelled{false};
            backend->fail = fail;
            backend->cancellation = &cancelled;
            BatchEvaluator evaluator(std::move(backend), 8, 1000);
            std::mt19937_64 random(3);
            Search search(config, evaluator, random);
            bool rejected = false;
            try {
                auto result = search.run(game, {200, false}, false, {8, 1.0, &cancelled});
                require(!fail && result.cancelled && result.completed_visits < 200);
                require(result.completed_visits == std::accumulate(result.visits.begin(), result.visits.end(), 0));
            } catch (const std::runtime_error& error) {
                require(fail && std::string(error.what()) == "injected search failure");
                rejected = true;
            }
            require(rejected == fail);
        }
        {
            Game almost_full(5);
            for (int action : {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 10, 13, 12, 15, 14, 17, 16, 19, 18, 20, 21, 22, 23})
                almost_full.play(action);
            BatchEvaluator evaluator(std::make_unique<Backend>(), 8, 1000);
            std::mt19937_64 random(3);
            Search search(config, evaluator, random);
            auto result = search.run(almost_full, {32, false}, false, {8, 1.0});
            require(result.completed_visits == 32 && result.visits[24] == 32);
            require(result.selection_policy[24] == 1);
            std::atomic<bool> cancelled{true};
            auto stopped = search.run(almost_full, {32, false}, false, {8, 1.0, &cancelled});
            require(stopped.cancelled && stopped.completed_visits == 0 && stopped.selection_policy[24] == 1);
        }
        Node root;
        root.update(0);
        for (int i = 0; i < 2; ++i) {
            auto child = std::make_unique<Node>();
            child->parent = &root;
            child->action = i;
            child->prior = 0.5;
            for (int n = 0; n < (i == 0 ? 100 : 50); ++n) child->update(i == 0 ? 0.8 : -0.8);
            root.children.push_back(std::move(child));
        }
        auto selected = search_result(root, config, 25);
        require(selected.visits[0] > selected.visits[1]);
        require(selected.selection_policy[1] > selected.selection_policy[0]);
        std::mt19937_64 random(1);
        require(choose_action(selected.selection_policy, 0, random) == 1);
        config.use_lcb = false;
        require(search_result(root, config, 25).selection_policy == selected.visit_policy);
        root.children[0]->value_sum = 0;
        root.children[1]->value_sum = 0;
        root.children[1]->visits = root.children[0]->visits;
        require(select_child(root, config, false, 1.0)->action == 0);
        root.children[0]->in_flight = 1;
        require(select_child(root, config, false, 1.0)->action == 1);
        std::cout << "parallel search invariants passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
