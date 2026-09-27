#include "muzero/eval_session.h"
#include <iostream>

using namespace muzero;

void require(bool condition) {
    if (!condition) throw std::runtime_error("session assertion failed");
}

struct State : Latent {};
struct Uniform : Evaluator {
    Evaluation initial(const std::vector<float>&) override {
        return {std::make_shared<State>(), std::vector<double>(49, 0), 0};
    }
    Evaluation recurrent(const std::shared_ptr<const Latent>&, int) override {
        return initial({});
    }
};

int main(int argc, char** argv) {
    try {
        require(argc == 2);
        Config raw(argv[1]);
        Uniform evaluator;
        EvalSession session(7, SearchConfig(raw), evaluator, {1, 1.0}, 1);
        session.reset(5, Rule::STANDARD);
        session.play(12);
        auto analysis = session.analyze(16);
        require(analysis.turn == 1 && analysis.player == -1);
        require(analysis.action != 12 && analysis.result.completed_visits == 16);
        require(session.game().turn() == 1);
        session.play(analysis.action);
        session.undo(2);
        require(session.game().turn() == 0 && session.moves().empty());
        bool rejected = false;
        try { session.undo(1); } catch (const std::exception&) { rejected = true; }
        require(rejected && session.game().turn() == 0);
        session.play(0);
        rejected = false;
        try { session.reset(8, Rule::STANDARD); } catch (const std::exception&) { rejected = true; }
        require(rejected && session.game().turn() == 1);
        session.reset(5, Rule::STANDARD);
        for (int move : {0, 5, 1, 6, 2, 7, 3, 8, 4}) session.play(move);
        require(session.game().finished() && session.game().winner() == 1);
        rejected = false;
        try { session.analyze(16); } catch (const std::exception&) { rejected = true; }
        require(rejected);
        session.undo(1);
        require(!session.game().finished() && session.game().turn() == 8);
        std::cout << "eval session invariants passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
