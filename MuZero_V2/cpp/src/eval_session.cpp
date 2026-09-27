#include "muzero/eval_session.h"
#include <chrono>

namespace muzero {

const Game& EvalSession::game() const {
    if (!game_) throw std::runtime_error("Start a game first");
    return *game_;
}

void EvalSession::reset(int size, Rule rule) {
    Game next(size, canvas_, rule);
    game_ = std::move(next);
    moves_.clear();
}

void EvalSession::play(int action) {
    int size = game().size();
    if (action < 0 || action >= size * size) throw std::runtime_error("Move outside board");
    game_->play(action / size * canvas_ + action % size);
    moves_.push_back(action);
}

void EvalSession::undo(int count) {
    if (count < 1 || count > static_cast<int>(moves_.size())) throw std::runtime_error("Invalid undo count");
    Game next(game().size(), canvas_, game().rule());
    size_t remaining = moves_.size() - count;
    for (size_t i = 0; i < remaining; ++i)
        next.play(moves_[i] / next.size() * canvas_ + moves_[i] % next.size());
    game_ = std::move(next);
    moves_.resize(remaining);
}

Analysis EvalSession::analyze(int visits) {
    if (visits < 1) throw std::runtime_error("Search visits must be positive");
    auto begin = std::chrono::steady_clock::now();
    Search search(config_, evaluator_, random_);
    auto result = search.run(game(), {visits, false}, false, execution_);
    double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
    int action = choose_action(result.selection_policy, config_.move_temperature(game().turn(), game().size()), random_);
    return {std::move(result), action / canvas_ * game().size() + action % canvas_, game().turn(), game().player(), seconds};
}

}
