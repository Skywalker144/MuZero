#include "muzero/match.h"
#include <chrono>

namespace muzero {

std::optional<OpeningResult> match_opening(int size, int canvas, Rule rule, const OpeningConfig& config,
                                         Evaluator& evaluator, uint64_t seed, const std::atomic<bool>& cancelled) {
    std::mt19937_64 random(seed);
    for (int retry = 0; retry < 100; ++retry) {
        if (cancelled) return std::nullopt;
        Game game(size, canvas, rule);
        auto result = initialize_opening(game, config, evaluator, random, [&] { return cancelled.load(); });
        if (result.status == OpeningStatus::Interrupted) return std::nullopt;
        if (result.status != OpeningStatus::Success)
            throw std::runtime_error("Balanced opening failed: " + result.failure);
        if (!game.finished()) return result;
    }
    throw std::runtime_error("100 consecutive terminal openings");
}

std::optional<MatchResult> play_match(Game game, const SearchConfig& config, Evaluator& a, Evaluator& b,
                                    bool black_a, SearchExecution execution, uint64_t seed) {
    if (game.finished()) throw std::runtime_error("Match opening is terminal");
    auto start = std::chrono::steady_clock::now();
    std::mt19937_64 random(seed);
    MatchResult match{};
    while (!game.finished()) {
        if (execution.cancelled && execution.cancelled->load()) return std::nullopt;
        Evaluator& evaluator = (game.player() == 1) == black_a ? a : b;
        Search search(config, evaluator, random);
        auto result = search.run(game, {config.full_search_visits, false}, false, execution);
        if (result.cancelled) return std::nullopt;
        int action = choose_action(result.selection_policy, config.move_temperature(game.turn(), game.size()), random);
        game.play(action);
        match.moves.push_back(action);
    }
    match.winner = game.winner();
    match.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return match;
}

}
