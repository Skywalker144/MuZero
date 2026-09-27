#include "muzero/record.h"

namespace muzero {

void apply_training_weights(FinishedGame& game, double policy_factor, double value_factor) {
    if (policy_factor < 0 || value_factor < 0 || policy_factor + value_factor > 1)
        throw std::runtime_error("Invalid surprise weighting");
    if (game.steps.empty() || (policy_factor == 0 && value_factor == 0)) return;
    std::array<double, 3> future{double(game.winner == 1), double(game.winner == 0), double(game.winner == -1)};
    std::vector<double> value_surprise(game.steps.size());
    double now = 1 / (1 + game.size * game.size * 0.016);
    for (size_t i = game.steps.size(); i-- > 0;) {
        const auto& step = game.steps[i];
        double kl = 0;
        for (int j = 0; j < 3; ++j) {
            int index = step.player == 1 ? j : 2 - j;
            future[j] += now * (step.search_wdl[index] - future[j]);
            if (future[j] > 1e-100)
                kl += future[j] * std::log(future[j] / std::max(1e-100, step.network_wdl[index]));
        }
        value_surprise[i] = std::clamp(kl, 0.0, 1.0);
    }
    double sum_weights = 0, sum_policy = 0, sum_value = 0;
    for (size_t i = 0; i < game.steps.size(); ++i) {
        const auto& step = game.steps[i];
        sum_weights += step.weight;
        sum_policy += step.weight * step.policy_surprise;
        sum_value += step.weight * value_surprise[i];
    }
    if (sum_weights < 1) return;
    double average_policy = sum_policy / sum_weights;
    double average_value = sum_value / sum_weights;
    value_factor *= std::min(1.0, average_value / 0.010);
    std::vector<double> policy_props(game.steps.size()), value_props(game.steps.size());
    sum_policy = sum_value = 0;
    for (size_t i = 0; i < game.steps.size(); ++i) {
        const auto& step = game.steps[i];
        double excess = std::max(0.0, step.policy_surprise - 1.5 * average_policy);
        policy_props[i] = step.weight * step.policy_surprise + (1 - step.weight) * excess;
        value_props[i] = step.weight * value_surprise[i];
        sum_policy += policy_props[i];
        sum_value += value_props[i];
    }
    for (size_t i = 0; i < game.steps.size(); ++i)
        game.steps[i].weight = (1 - policy_factor - value_factor) * game.steps[i].weight +
            policy_factor * policy_props[i] * sum_weights / std::max(sum_policy, 1e-10) +
            value_factor * value_props[i] * sum_weights / std::max(sum_value, 1e-10);
}

}
