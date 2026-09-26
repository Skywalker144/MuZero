#include "muzero/search.h"
#include "muzero/game.h"
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace muzero;

void require(bool condition) {
    if (!condition) throw std::runtime_error("test assertion failed");
}

struct DummyLatent : Latent {};

struct UniformEvaluator : Evaluator {
    int actions;
    int initial_calls = 0;
    int recurrent_calls = 0;
    int active_size = 0, canvas = 0;
    explicit UniformEvaluator(int count) : actions(count) {}
    Evaluation initial(const std::vector<float>& observation) override {
        require(observation.size() == static_cast<size_t>(INPUT_PLANES * actions));
        ++initial_calls;
        return {std::make_shared<DummyLatent>(), std::vector<double>(actions, 0.0), 0.5};
    }
    Evaluation recurrent(const std::shared_ptr<const Latent>& hidden, int action) override {
        require(hidden != nullptr && action >= 0 && action < actions);
        if (active_size) require(action / canvas < active_size && action % canvas < active_size);
        ++recurrent_calls;
        return {std::make_shared<DummyLatent>(), std::vector<double>(actions, 0.0), 0.5};
    }
};

int main(int argc, char** argv) {
    try {
        require(argc == 2);
        Config raw(argv[1]);
        SearchConfig config(raw);
        Game game(5);
        for (int action : {0, 5, 1, 6, 2, 7, 3, 8}) game.play(action);
        require(!game.finished());
        auto observation = game.observation();
        require(observation[0] == 1.0f && observation[25 + 5] == 1.0f);
        game.play(4);
        require(game.finished() && game.winner() == 1);
        bool rejected = false;
        try { game.play(5); } catch (const std::exception&) { rejected = true; }
        require(rejected);
        Game draw(5);
        for (int action : {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 10, 13, 12, 15, 14, 17, 16, 19, 18, 20, 21, 22, 23, 24}) draw.play(action);
        require(draw.finished() && draw.winner() == 0);
        Game overline(6);
        for (int action : {0, 6, 1, 8, 3, 10, 4, 12, 5, 14, 2}) overline.play(action);
        require(overline.finished() && overline.winner() == 1);
        Game root(5);
        root.play(12);
        UniformEvaluator evaluator(25);
        std::mt19937_64 random(1);
        Search search(config, evaluator, random);
        auto result = search.run(root, {32, false}, false);
        require(evaluator.initial_calls == 1 && evaluator.recurrent_calls == 32);
        require(result.visits[12] == 0 && result.policy_target[12] == 0);
        require(std::abs(std::accumulate(result.visit_policy.begin(), result.visit_policy.end(), 0.0) - 1) < 1e-9);
        require(std::accumulate(result.visits.begin(), result.visits.end(), 0) == 32);
        auto direct = search.run(root, {0, false}, false);
        require(evaluator.recurrent_calls == 32 && direct.visit_policy[12] == 0);
        auto single = search.run(root, {1, false}, false);
        require(std::abs(single.root_value) < 1e-12);
        {
            Game padded(5, 9, Rule::STANDARD);
            padded.play(0);
            UniformEvaluator mixed(81);
            mixed.active_size = 5;
            mixed.canvas = 9;
            Search padded_search(config, mixed, random);
            auto padded_result = padded_search.run(padded, {32, false}, false);
            for (int action = 0; action < 81; ++action)
                if (!padded.legal(action)) require(padded_result.visit_policy[action] == 0);
            require(mixed.recurrent_calls == 32);
        }
        Node parent;
        parent.nn_value = 0.5;
        parent.update(0.5);
        parent.children.emplace_back(std::make_unique<Node>());
        parent.children[0]->prior = 1;
        parent.children[0]->parent = &parent;
        parent.children[0]->action = 2;
        parent.children[0]->update(-0.75);
        require(select_child(parent, config, false)->action == 2);
        Node selection;
        selection.update(0.5);
        selection.nn_value = 1.0;
        for (int i = 0; i < 2; ++i) {
            auto child = std::make_unique<Node>();
            child->action = i;
            child->prior = 0.0;
            child->parent = &selection;
            selection.children.push_back(std::move(child));
        }
        selection.children[0]->update(0.5);
        config.use_fpu = false;
        require(select_child(selection, config, false)->action == 0);
        config.use_fpu = true;
        require(select_child(selection, config, false)->action == 1);
        config.move_schedule = "threshold";
        config.move_early = 1.0;
        config.move_late = 0.0;
        require(config.move_temperature(config.temperature_moves - 1, 5) == 1.0);
        require(config.move_temperature(config.temperature_moves, 5) == 0.0);
        std::cout << "core invariants passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
