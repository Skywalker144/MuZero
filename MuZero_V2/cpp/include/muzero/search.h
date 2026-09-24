#pragma once

#include "config.h"
#include "game.h"
#include <algorithm>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

namespace muzero {

struct Latent { virtual ~Latent() = default; };
struct Evaluation {
    std::shared_ptr<const Latent> hidden;
    std::vector<double> logits;
    double value;
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
    double prior = 0, value_sum = 0, value_sq_sum = 0, nn_value = 0;
    std::shared_ptr<const Latent> hidden;
    std::vector<std::unique_ptr<Node>> children;
    void update(double value) { ++visits; value_sum += value; value_sq_sum += value * value; }
    double mean() const { return visits ? value_sum / visits : 0; }
};
struct SearchResult {
    std::vector<double> visit_policy, policy_target;
    std::vector<int> visits;
    double root_value;
};

Node* select_child(Node& node, const SearchConfig& config, bool cheap);
std::vector<double> softmax(const std::vector<double>& logits);
int choose_action(const std::vector<double>& policy, double temperature, std::mt19937_64& random);

class Search {
    const SearchConfig& config_;
    Evaluator& evaluator_;
    std::mt19937_64& random_;
public:
    Search(const SearchConfig& config, Evaluator& evaluator, std::mt19937_64& random)
        : config_(config), evaluator_(evaluator), random_(random) {}
    SearchResult run(const Game& game, Budget budget, bool training);
};

}
