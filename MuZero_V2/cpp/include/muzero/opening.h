#pragma once

#include "search.h"
#include <functional>

namespace muzero {

struct OpeningConfig {
    double probability, avg_dist_factor, balance_exponent, rejection_probability, rejection_probability_fallback;
    int max_tries;
    bool policy_init, policy_after, policy_on_failure;
    double policy_init_mean, policy_temperature;
    explicit OpeningConfig(const Config& c);
};

enum class OpeningStatus { NotAttempted, Success, Failed, Interrupted };

struct OpeningResult {
    OpeningStatus status = OpeningStatus::NotAttempted;
    int attempts = 0, balanced_moves = 0, policy_moves = 0;
    double start_value = 0;
    std::string failure;
    std::vector<int> actions;
};

OpeningResult initialize_opening(Game& game, const OpeningConfig& config, Evaluator& evaluator,
                                 std::mt19937_64& random, const std::function<bool()>& cancelled = {});

}
