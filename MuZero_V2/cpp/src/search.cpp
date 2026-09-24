#include "muzero/search.h"
#include <limits>

namespace muzero {

std::vector<double> softmax(const std::vector<double>& logits) {
    double maximum = *std::max_element(logits.begin(), logits.end());
    if (!std::isfinite(maximum)) throw std::runtime_error("No finite policy logits");
    std::vector<double> result(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) result[i] = std::exp(logits[i] - maximum);
    double mass = std::accumulate(result.begin(), result.end(), 0.0);
    if (!std::isfinite(mass) || mass <= 0) throw std::runtime_error("Invalid policy mass");
    for (auto& weight : result) weight /= mass;
    return result;
}

Node* select_child(Node& node, const SearchConfig& c, bool cheap) {
    double mass = 0;
    if (c.use_fpu)
        for (const auto& child : node.children) if (child->visits) mass += child->prior;
    mass = std::min(1.0, mass);
    double mix = std::pow(mass, c.fpu_power);
    double reduction = !node.parent && !cheap ? c.root_fpu : c.fpu;
    double fpu = mix * node.mean() + (1 - mix) * node.nn_value - reduction * std::sqrt(mass);
    double pb_c = (c.pb_c_init + std::log((node.visits + c.pb_c_base + 1) / c.pb_c_base)) * std::sqrt(node.visits);
    Node* best = nullptr;
    double best_score = -std::numeric_limits<double>::infinity();
    for (const auto& child : node.children) {
        double value_score = child->visits ? (1 - child->mean()) / 2 : c.use_fpu ? (fpu + 1) / 2 : 0;
        double score = value_score + pb_c * child->prior / (child->visits + 1);
        if (score > best_score) { best_score = score; best = child.get(); }
    }
    return best;
}

int choose_action(const std::vector<double>& policy, double temperature, std::mt19937_64& random) {
    if (temperature == 0)
        return static_cast<int>(std::max_element(policy.begin(), policy.end()) - policy.begin());
    std::vector<double> logits(policy.size());
    double maximum = *std::max_element(policy.begin(), policy.end());
    for (size_t i = 0; i < policy.size(); ++i)
        logits[i] = policy[i] > 0 ? (std::log(policy[i]) - std::log(maximum)) / temperature : -std::numeric_limits<double>::infinity();
    auto weights = softmax(logits);
    return std::discrete_distribution<int>(weights.begin(), weights.end())(random);
}

SearchResult Search::run(const Game& game, Budget budget, bool training) {
    if (game.finished() || budget.visits < 0 || (training && budget.visits == 0))
        throw std::runtime_error("Invalid search root or budget");
    Node root;
    const auto& c = config_;
    for (int visit = 0; visit <= budget.visits; ++visit) {
        Node* node = &root;
        while (!node->children.empty()) node = select_child(*node, c, budget.cheap);
        auto evaluation = node->parent ? evaluator_.recurrent(node->parent->hidden, node->action)
                                       : evaluator_.initial(game.observation());
        if (evaluation.logits.size() != static_cast<size_t>(game.actions()) ||
            !std::isfinite(evaluation.value) || evaluation.value < -1.00001 || evaluation.value > 1.00001)
            throw std::runtime_error("Invalid evaluator output");
        for (double logit : evaluation.logits)
            if (!std::isfinite(logit)) throw std::runtime_error("Nonfinite network logit");
        node->hidden = std::move(evaluation.hidden);
        node->nn_value = evaluation.value;
        bool noisy = node == &root && training && !budget.cheap;
        std::vector<int> actions;
        for (int action = 0; action < game.actions(); ++action) {
            if (node == &root && !game.legal(action))
                evaluation.logits[action] = -std::numeric_limits<double>::infinity();
            else actions.push_back(action);
        }
        if (noisy)
            for (auto& logit : evaluation.logits)
                logit /= c.temperature(c.root_early, c.root_late, game.turn(), game.size());
        auto policy = softmax(evaluation.logits);
        if (noisy && c.noise_weight > 0 && actions.size() > 1) {
            std::vector<double> proportions(actions.size(), 1.0 / actions.size());
            if (c.shaped_noise) {
                std::vector<double> shape(actions.size());
                double cap = 0.01 * std::pow(19.0 / game.size(), 2);
                for (size_t i = 0; i < actions.size(); ++i)
                    shape[i] = std::log(std::min(cap, policy[actions[i]]) + 1e-20);
                double mean = std::accumulate(shape.begin(), shape.end(), 0.0) / shape.size();
                for (auto& value : shape) value = std::max(0.0, value - mean);
                double mass = std::accumulate(shape.begin(), shape.end(), 0.0);
                if (mass > 0)
                    for (size_t i = 0; i < shape.size(); ++i)
                        proportions[i] = 0.5 * (proportions[i] + shape[i] / mass);
            }
            std::vector<double> noise(actions.size());
            double mass = 0;
            do {
                for (size_t i = 0; i < noise.size(); ++i)
                    noise[i] = std::gamma_distribution<double>(proportions[i] * c.concentration, 1.0)(random_);
                mass = std::accumulate(noise.begin(), noise.end(), 0.0);
            } while (mass == 0);
            for (size_t i = 0; i < actions.size(); ++i)
                policy[actions[i]] = (1 - c.noise_weight) * policy[actions[i]] + c.noise_weight * noise[i] / mass;
        }
        for (int action : actions) {
            auto child = std::make_unique<Node>();
            child->parent = node;
            child->action = action;
            child->prior = policy[action];
            node->children.push_back(std::move(child));
        }
        double value = evaluation.value;
        for (Node* ancestor = node; ancestor; ancestor = ancestor->parent) {
            ancestor->update(value);
            value = -value;
        }
    }
    SearchResult result;
    result.visit_policy.resize(game.actions());
    result.visits.resize(game.actions());
    for (const auto& child : root.children) {
        result.visits[child->action] = child->visits;
        result.visit_policy[child->action] = budget.visits ? child->visits : child->prior;
    }
    result.policy_target = result.visit_policy;
    if (c.use_lcb && budget.visits && !budget.cheap) {
        std::vector<double> radius(root.children.size(), 2 * c.lcb_stdevs), lcb(root.children.size(), -2 * c.lcb_stdevs);
        const Node* stable = root.children.front().get();
        auto stability = [](const Node& n) { return std::max(0, n.visits - 1) + 2 * n.prior; };
        for (size_t i = 0; i < root.children.size(); ++i) {
            const auto& child = *root.children[i];
            if (stability(child) > stability(*stable)) stable = &child;
            if (!child.visits) continue;
            double n = child.visits, mean = child.mean(), prior_weight = 1 / (n * n);
            double variance = std::max(0.0, child.value_sq_sum / n - mean * mean) + prior_weight / (n + prior_weight);
            double effective = std::pow(n + prior_weight, 2) / (n + prior_weight * prior_weight);
            radius[i] = c.lcb_stdevs * std::sqrt(variance / effective);
            lcb[i] = -mean - radius[i];
        }
        int best = -1;
        for (size_t i = 0; i < root.children.size(); ++i) {
            const auto& child = *root.children[i];
            if (child.visits > 0 && child.visits >= c.min_lcb_visits * stable->visits &&
                (best < 0 || lcb[i] > lcb[best])) best = static_cast<int>(i);
        }
        if (best < 0) throw std::runtime_error("No LCB candidate");
        int action = root.children[best]->action;
        for (size_t i = 0; i < root.children.size(); ++i) {
            const auto& child = *root.children[i];
            double excess = lcb[best] - lcb[i];
            if (static_cast<int>(i) == best || !child.visits || excess <= 0) continue;
            double factor = (radius[i] + excess) / (radius[i] + 0.20 * excess);
            result.policy_target[action] = std::max(result.policy_target[action], factor * factor * child.visits);
        }
    }
    for (auto* policy : {&result.visit_policy, &result.policy_target}) {
        double mass = std::accumulate(policy->begin(), policy->end(), 0.0);
        for (auto& weight : *policy) weight /= mass;
    }
    result.root_value = root.mean();
    return result;
}

}
