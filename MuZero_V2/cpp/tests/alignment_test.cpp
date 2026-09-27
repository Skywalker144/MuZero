#include "muzero/record.h"
#include <iostream>

using namespace muzero;

void require(bool condition) {
    if (!condition) throw std::runtime_error("Alignment assertion failed");
}

class ConcentratedEvaluator : public Evaluator {
    int preferred_;
public:
    explicit ConcentratedEvaluator(int preferred) : preferred_(preferred) {}
    Evaluation evaluate() {
        std::vector<double> logits(225, -20);
        logits[preferred_] = 0;
        return {std::make_shared<Latent>(), std::move(logits), 0.0};
    }
    Evaluation initial(const std::vector<float>&) override { return evaluate(); }
    Evaluation recurrent(const std::shared_ptr<const Latent>&, int) override { return evaluate(); }
};

int main(int argc, char** argv) {
    try {
        require(argc == 2);
        Config raw(argv[1]);
        SearchConfig c(raw);
        for (int preferred : {0, 224}) {
            ConcentratedEvaluator evaluator(preferred);
            SearchConfig search_config(raw);
            search_config.noise_weight = 0;
            search_config.root_desired_visits = 2;
            for (int visits : {80, 400}) {
                std::mt19937_64 random(1);
                Search search(search_config, evaluator, random);
                auto result = search.run(Game(15), {visits, false}, true);
                require(result.completed_visits == visits);
                require(result.visits[preferred] == visits);
                require(std::count_if(result.visits.begin(), result.visits.end(), [](int n) { return n > 0; }) == 1);
            }
        }
        auto alpha = dirichlet_alpha_distribution({0.5, 0.25, 0.1, 0.05, 0.04, 0.03, 0.02, 0.009, 0.001, -1});
        require(alpha.back() == 0);
        require(std::abs(alpha[0] - alpha[6]) < 1e-12);
        require(std::abs(std::accumulate(alpha.begin(), alpha.end(), 0.0) - 1) < 1e-12);
        Node root;
        root.update(-1);
        root.nn_value = -1;
        for (int i = 0; i < 2; ++i) {
            auto child = std::make_unique<Node>();
            child->parent = &root;
            child->action = i;
            child->prior = 0.5;
            root.children.push_back(std::move(child));
        }
        for (int i = 0; i < 100; ++i) root.children[0]->update(-0.8);
        c.pb_c_init = 0;
        c.pb_c_base = 1e30;
        c.root_desired_visits = 2;
        require(select_child(root, c, false)->action == 0);
        root.children[1]->update(0.8);
        require(select_child(root, c, false)->action == 1);
        require(select_child(root, c, true)->action == 0);
        root.children[1]->in_flight = 10;
        require(select_child(root, c, false)->action == 0);
        root.children[1]->in_flight = 0;
        c.root_desired_visits = 0;
        require(select_child(root, c, false)->action == 0);
        c.root_desired_visits = 2;
        for (int i = 1; i < 60; ++i) root.children[1]->update(0.8);
        c.pb_c_init = 1.25;
        c.use_lcb = false;
        c.policy_target_pruning = true;
        root.visits = 161;
        auto result = search_result(root, c, 2);
        double explore = 1.25 * std::sqrt(161.0);
        double best = 0.9 + explore * 0.5 / 101;
        double reduced = std::ceil(explore * 0.5 / (best - 0.1) - 1);
        require(std::abs(result.policy_target[1] - reduced / (100 + reduced)) < 1e-12);
        require(result.move_policy == result.policy_target);
        c.policy_target_pruning = false;
        auto unpruned = search_result(root, c, 2);
        require(std::abs(unpruned.policy_target[1] - 60.0 / 160) < 1e-12);
        FinishedGame game{0, 5, 5, Rule::FREESTYLE, 1, {}, {}};
        game.steps.resize(2);
        game.steps[0].player = 1;
        game.steps[1].player = -1;
        game.steps[0].weight = 1;
        game.steps[1].weight = 0;
        game.steps[0].policy_surprise = 0.2;
        game.steps[1].policy_surprise = 0.8;
        apply_training_weights(game, 0.5, 0);
        require(std::abs(game.steps[0].weight - (0.5 + 0.5 * 0.2 / 0.7)) < 1e-12);
        require(std::abs(game.steps[1].weight - 0.5 * 0.5 / 0.7) < 1e-12);
        std::cout << "Search and training-weight alignment passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
