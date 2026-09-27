#pragma once

#include "config.h"
#include "game.h"
#include <algorithm>
#include <atomic>
#include <array>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

namespace muzero {

struct Latent { virtual ~Latent() = default; };
struct Evaluation {
    std::shared_ptr<const Latent> hidden;
    std::vector<double> logits;
    std::array<double, 3> wdl{0.5, 0, 0.5};
    Evaluation() = default;
    Evaluation(std::shared_ptr<const Latent> state, std::vector<double> policy, double utility)
        : hidden(std::move(state)), logits(std::move(policy)), wdl{(1 + utility) / 2, 0, (1 - utility) / 2} {}
    double value() const { return wdl[0] - wdl[2]; }
};
class Evaluator {
public:
    virtual ~Evaluator() = default;
    virtual Evaluation initial(const std::vector<float>& observation) = 0;
    virtual Evaluation recurrent(const std::shared_ptr<const Latent>& hidden, int action) = 0;
};
struct Budget { int visits; bool cheap; };
struct Node {
    Node* parent = nullptr;
    int action = -1, visits = 0;
    int in_flight = 0;
    bool expanding = false;
    double prior = 0, value_sum = 0, value_sq_sum = 0, nn_value = 0;
    std::array<double, 3> wdl_sum{};
    std::shared_ptr<const Latent> hidden;
    std::vector<std::unique_ptr<Node>> children;
    void update(const std::array<double, 3>& wdl) {
        double value = wdl[0] - wdl[2];
        ++visits;
        value_sum += value;
        value_sq_sum += value * value;
        for (size_t i = 0; i < wdl.size(); ++i) wdl_sum[i] += wdl[i];
    }
    void update(double value) { update({(1 + value) / 2, 0, (1 - value) / 2}); }
    double mean() const { return visits ? value_sum / visits : 0; }
};
struct SearchResult {
    std::vector<double> network_policy, visit_policy, move_policy, selection_policy, policy_target;
    std::array<double, 3> network_wdl, search_wdl;
    double policy_surprise = 0;
    std::vector<int> visits;
    double root_value;
    int completed_visits = 0;
    bool cancelled = false;
};
struct SearchExecution {
    int threads = 1;
    double virtual_loss = 1.0;
    const std::atomic<bool>* cancelled = nullptr;
};

Node* select_child(Node& node, const SearchConfig& config, bool cheap, double virtual_loss = 1.0);
SearchResult search_result(const Node& root, const SearchConfig& config, int actions);
std::vector<double> softmax(const std::vector<double>& logits);
std::vector<double> dirichlet_alpha_distribution(const std::vector<double>& policy);
int choose_action(const std::vector<double>& policy, double temperature, std::mt19937_64& random);

class Search {
    const SearchConfig& config_;
    Evaluator& evaluator_;
    std::mt19937_64& random_;
public:
    Search(const SearchConfig& config, Evaluator& evaluator, std::mt19937_64& random)
        : config_(config), evaluator_(evaluator), random_(random) {}
    SearchResult run(const Game& game, Budget budget, bool training, SearchExecution execution = {});
};

}
