#include "muzero/random_evaluator.h"

namespace muzero {
namespace {

uint64_t mix(uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

struct RandomLatent : Latent {
    uint64_t key;
    explicit RandomLatent(uint64_t value) : key(value) {}
};

}

RandomEvaluator::RandomEvaluator(int canvas, uint64_t seed) : actions_(canvas * canvas), seed_(mix(seed)) {
    if (canvas < 5 || canvas > 25) throw std::runtime_error("Invalid random evaluator canvas");
}

Evaluation RandomEvaluator::evaluate(uint64_t key) const {
    std::mt19937_64 random(key);
    std::normal_distribution<double> gaussian;
    std::vector<double> logits(actions_);
    for (double& logit : logits) logit = gaussian(random);
    return {std::make_shared<RandomLatent>(key), std::move(logits), std::tanh(0.2 * gaussian(random))};
}

Evaluation RandomEvaluator::initial(const std::vector<float>& observation) {
    if (observation.size() != static_cast<size_t>(INPUT_PLANES * actions_))
        throw std::runtime_error("Observation shape mismatch");
    uint64_t key = seed_;
    for (float value : observation) {
        if (value != 0 && value != 1) throw std::runtime_error("Invalid binary observation");
        key = mix(key ^ static_cast<uint64_t>(value));
    }
    return evaluate(key);
}

Evaluation RandomEvaluator::recurrent(const std::shared_ptr<const Latent>& hidden, int action) {
    auto state = std::dynamic_pointer_cast<const RandomLatent>(hidden);
    if (!state || action < 0 || action >= actions_) throw std::runtime_error("Invalid random recurrent input");
    return evaluate(mix(state->key ^ mix(static_cast<uint64_t>(action))));
}

}
