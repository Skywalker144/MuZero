#include "muzero/random_evaluator.h"
#include <iostream>

using namespace muzero;

void require(bool condition) {
    if (!condition) throw std::runtime_error("test assertion failed");
}

int main(int argc, char** argv) {
    try {
        require(argc == 2);
        Config raw(argv[1]);
        SearchConfig config(raw);
        RandomEvaluator evaluator(9, 42), replica(9, 42), other(9, 43);
        Game game(5, 9, Rule::RENJU);
        game.play(0);
        auto first = evaluator.initial(game.observation());
        auto repeated = replica.initial(game.observation());
        require(first.hidden && first.logits.size() == 81);
        require(first.logits == repeated.logits && first.value() == repeated.value());
        require(first.logits != other.initial(game.observation()).logits);
        auto child = evaluator.recurrent(first.hidden, 1);
        auto same_child = replica.recurrent(repeated.hidden, 1);
        require(child.logits == same_child.logits && child.value() == same_child.value());
        require(child.logits != evaluator.recurrent(first.hidden, 2).logits);
        require(child.logits != evaluator.recurrent(child.hidden, 1).logits);
        double total = 0;
        for (int seed = 0; seed < 1000; ++seed) {
            RandomEvaluator sample(9, seed);
            auto result = sample.initial(game.observation());
            require(std::isfinite(result.value()) && std::abs(result.value()) <= 1);
            for (double logit : result.logits) require(std::isfinite(logit));
            total += result.value();
        }
        require(std::abs(total / 1000) < 0.03);
        std::mt19937_64 random(17), matching_random(17);
        Search search(config, evaluator, random), matching_search(config, replica, matching_random);
        auto result = search.run(game, {32, false}, true);
        auto matching = matching_search.run(game, {32, false}, true);
        require(result.policy_target == matching.policy_target);
        require(result.visits == matching.visits);
        require(std::accumulate(result.visits.begin(), result.visits.end(), 0) == 32);
        require(std::abs(std::accumulate(result.policy_target.begin(), result.policy_target.end(), 0.0) - 1) < 1e-9);
        for (int action = 0; action < game.actions(); ++action)
            if (!game.legal(action)) require(result.policy_target[action] == 0);
        bool rejected = false;
        try { evaluator.recurrent(std::make_shared<Latent>(), 1); }
        catch (const std::exception&) { rejected = true; }
        require(rejected);
        std::cout << "random evaluator invariants passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
