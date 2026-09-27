#pragma once

#include "search.h"
#include <optional>

namespace muzero {

struct Analysis {
    SearchResult result;
    int action, turn, player;
    double seconds;
};

class EvalSession {
    int canvas_;
    SearchConfig config_;
    Evaluator& evaluator_;
    SearchExecution execution_;
    std::mt19937_64 random_;
    std::optional<Game> game_;
    std::vector<int> moves_;
public:
    EvalSession(int canvas, SearchConfig config, Evaluator& evaluator, SearchExecution execution, uint64_t seed)
        : canvas_(canvas), config_(std::move(config)), evaluator_(evaluator), execution_(execution), random_(seed) {}
    const Game& game() const;
    const std::vector<int>& moves() const { return moves_; }
    void reset(int size, Rule rule);
    void play(int action);
    void undo(int count);
    Analysis analyze(int visits);
};

}
