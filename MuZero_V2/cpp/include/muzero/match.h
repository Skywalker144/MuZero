#pragma once

#include "opening.h"
#include <optional>

namespace muzero {

struct MatchResult {
    int winner;
    std::vector<int> moves;
    double seconds;
};

std::optional<OpeningResult> match_opening(int size, int canvas, Rule rule, const OpeningConfig& config,
                                         Evaluator& evaluator, uint64_t seed, const std::atomic<bool>& cancelled);
std::optional<MatchResult> play_match(Game game, const SearchConfig& config, Evaluator& a, Evaluator& b,
                                    bool black_a, SearchExecution execution, uint64_t seed);

}
