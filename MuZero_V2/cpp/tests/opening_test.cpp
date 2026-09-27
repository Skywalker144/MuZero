#include "muzero/opening.h"
#include <iostream>
#include <limits>

using namespace muzero;

void require(bool condition) {
    if (!condition) throw std::runtime_error("Opening assertion failed");
}

struct OpeningEvaluator : Evaluator {
    int canvas, calls = 0;
    std::vector<std::vector<float>> observations;
    double value = 0;
    bool record = false, force_line = false;
    int forced_action = -1;
    explicit OpeningEvaluator(int size) : canvas(size) {}
    Evaluation initial(const std::vector<float>& obs) override {
        ++calls;
        if (record) observations.push_back(obs);
        int n = canvas * canvas;
        std::vector<double> logits(n, 0);
        if (force_line) {
            std::fill(logits.begin(), logits.end(), -1000);
            int stones = 0;
            for (int a = 0; a < n; ++a) stones += obs[a] + obs[n + a];
            int moves[] = {0, 5, 1, 6, 2, 7, 3, 8, 4};
            require(stones < 9);
            logits[moves[stones]] = 0;
        }
        if (forced_action >= 0) {
            std::fill(logits.begin(), logits.end(), -1000);
            logits[forced_action] = 0;
        }
        return {nullptr, std::move(logits), value};
    }
    Evaluation recurrent(const std::shared_ptr<const Latent>&, int) override {
        throw std::runtime_error("Opening must never use dynamics");
    }
};

int main(int argc, char** argv) {
    try {
        require(argc == 2);
        OpeningConfig config{Config(argv[1])};
        config.probability = 1;
        config.policy_init = false;
        config.rejection_probability = 0;
        {
            Game game(5, 7, Rule::RENJU);
            game.play(0);
            game.play(7);
            game.play(1);
            auto actual = game.observation(), opposite = game.observation(-game.player());
            int n = game.actions();
            for (int a = 0; a < n; ++a) {
                require(actual[a] == opposite[n + a]);
                require(actual[n + a] == opposite[a]);
                require(opposite[static_cast<int>(Plane::BLACK_TO_MOVE) * n + a] == game.on_board(a));
                require(actual[static_cast<int>(Plane::FORBIDDEN_WHITE_TURN) * n + a] ==
                        opposite[static_cast<int>(Plane::FORBIDDEN_BLACK_TURN) * n + a]);
            }
            require(game.turn() == 3 && game.player() == -1);
        }
        for (Rule rule : {Rule::FREESTYLE, Rule::STANDARD, Rule::RENJU}) {
            std::mt19937_64 random_a(71), random_b(71);
            OpeningEvaluator evaluator(7);
            Game a(5, 7, rule), b = a;
            auto ra = initialize_opening(a, config, evaluator, random_a);
            auto rb = initialize_opening(b, config, evaluator, random_b);
            require(ra.status == OpeningStatus::Success && rb.status == ra.status);
            require(ra.actions == rb.actions && a.observation() == b.observation());
            require(a.turn() >= 1 && a.turn() <= 10 && !a.finished());
            Game replay(5, 7, rule);
            for (int action : ra.actions) replay.play(action);
            require(replay.observation() == a.observation());
        }
        {
            std::mt19937_64 random(12);
            OpeningEvaluator evaluator(5);
            evaluator.record = true;
            Game game(5);
            auto result = initialize_opening(game, config, evaluator, random);
            require(result.status == OpeningStatus::Success && evaluator.observations.size() >= 3);
            const auto& a = evaluator.observations[0];
            const auto& b = evaluator.observations[1];
            for (int i = 0; i < 25; ++i) {
                require(a[i] == b[25 + i] && b[i] == a[25 + i]);
                require(a[50 + i] + b[50 + i] == 1);
            }
        }
        {
            auto c = config;
            c.rejection_probability = 1;
            c.rejection_probability_fallback = 0;
            c.max_tries = 20;
            OpeningEvaluator evaluator(5);
            evaluator.value = 0.999;
            std::mt19937_64 random(10);
            Game game(5);
            auto result = initialize_opening(game, c, evaluator, random);
            require(result.status == OpeningStatus::Success && result.attempts == 22);
        }
        {
            OpeningEvaluator evaluator(5);
            evaluator.value = std::numeric_limits<double>::quiet_NaN();
            std::mt19937_64 random(9);
            Game game(5);
            auto result = initialize_opening(game, config, evaluator, random);
            require(result.status == OpeningStatus::Failed && result.actions.empty() && game.turn() == 0);
            require(result.failure == "nonfinite root evaluation");
        }
        {
            auto c = config;
            c.probability = 0;
            c.policy_init = true;
            c.policy_init_mean = 100;
            OpeningEvaluator evaluator(5);
            evaluator.force_line = true;
            std::mt19937_64 random(42);
            Game game(5);
            auto result = initialize_opening(game, c, evaluator, random);
            require(game.finished() && game.winner() == 1);
            require(result.policy_moves == 9 && result.actions.size() == 9 && evaluator.calls == 9);
        }
        {
            auto c = config;
            c.probability = 0;
            c.policy_init = true;
            c.policy_init_mean = 100;
            int terminal_games = 0;
            for (int seed = 0; seed < 12; ++seed) {
                Game game(15, 15, Rule::RENJU);
                for (int action : {111, 0, 113, 2, 97, 4, 127, 6}) game.play(action);
                OpeningEvaluator evaluator(15);
                evaluator.forced_action = 112;
                std::mt19937_64 random(seed);
                auto result = initialize_opening(game, c, evaluator, random);
                if (result.policy_moves == 0) continue;
                require(result.policy_moves == 1 && game.finished() && game.winner() == -1);
                require(result.actions == std::vector<int>{112});
                ++terminal_games;
            }
            require(terminal_games > 0);
        }
        {
            std::mt19937_64 random(5);
            OpeningEvaluator evaluator(5);
            Game game(5);
            auto result = initialize_opening(game, config, evaluator, random, [] { return true; });
            require(result.status == OpeningStatus::Interrupted && game.turn() == 0 && evaluator.calls == 0);
        }
        std::cout << "Opening tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
