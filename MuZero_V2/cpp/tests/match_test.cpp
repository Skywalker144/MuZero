#include "muzero/match.h"
#include <iostream>

using namespace muzero;

void require(bool value) {
    if (!value) throw std::runtime_error("match assertion failed");
}

struct State : Latent {
    const void* owner;
    explicit State(const void* p) : owner(p) {}
};

struct Counting : Evaluator {
    int calls = 0, color = 0;
    Evaluation initial(const std::vector<float>& obs) override {
        ++calls;
        if (color) require(obs[static_cast<int>(Plane::BLACK_TO_MOVE) * 25] == (color == 1));
        return {std::make_shared<State>(this), std::vector<double>(25, 0), 0};
    }
    Evaluation recurrent(const std::shared_ptr<const Latent>& state, int) override {
        require(std::dynamic_pointer_cast<const State>(state)->owner == this);
        return {std::make_shared<State>(this), std::vector<double>(25, 0), 0};
    }
};

int main(int argc, char** argv) {
    try {
        require(argc == 2);
        Config raw(argv[1]);
        SearchConfig search(raw);
        search.full_search_visits = 4;
        search.move_early = search.move_late = 0;
        OpeningConfig opening(raw);
        opening.probability = 1;
        opening.policy_init = false;
        Counting a, b;
        std::atomic<bool> stop{false};
        auto generated = match_opening(5, 5, Rule::RENJU, opening, a, 71, stop);
        require(generated.has_value() && !generated->actions.empty() && b.calls == 0);
        for (bool black_a : {true, false}) {
            a.color = black_a ? 1 : -1;
            b.color = -a.color;
            Game game(5, 5, Rule::RENJU);
            game.play(12);
            a.calls = b.calls = 0;
            auto result = play_match(game, search, a, b, black_a, {1, 1.0, &stop}, 19);
            require(result.has_value() && a.calls > 0 && b.calls > 0);
            Game replay(5, 5, Rule::RENJU);
            replay.play(12);
            for (int move : result->moves) replay.play(move);
            require(replay.finished() && replay.winner() == result->winner);
        }
        stop = true;
        require(!play_match(Game(5), search, a, b, true, {1, 1.0, &stop}, 19));
        require(!match_opening(5, 5, Rule::RENJU, opening, a, 71, stop));
        std::cout << "match routing, replay and cancellation passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
