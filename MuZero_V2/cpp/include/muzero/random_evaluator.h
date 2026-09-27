#pragma once

#include "search.h"
#include <cstdint>

namespace muzero {

class RandomEvaluator : public Evaluator {
    int actions_;
    uint64_t seed_;
    Evaluation evaluate(uint64_t key) const;
public:
    RandomEvaluator(int canvas, uint64_t seed);
    Evaluation initial(const std::vector<float>& observation) override;
    Evaluation recurrent(const std::shared_ptr<const Latent>& hidden, int action) override;
};

}
