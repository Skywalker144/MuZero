#include "muzero/search.h"
#include <limits>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>

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

std::vector<double> dirichlet_alpha_distribution(const std::vector<double>& policy) {
    std::vector<double> shape(policy.size(), 0.0);
    double count = 0, mean = 0;
    for (size_t i = 0; i < policy.size(); ++i) if (policy[i] >= 0) {
        shape[i] = std::log(std::min(0.01, policy[i]) + 1e-20);
        mean += shape[i];
        ++count;
    }
    if (count == 0) throw std::runtime_error("Empty noise policy");
    mean /= count;
    double mass = 0;
    for (size_t i = 0; i < policy.size(); ++i) if (policy[i] >= 0) {
        shape[i] = std::max(0.0, shape[i] - mean);
        mass += shape[i];
    }
    for (size_t i = 0; i < policy.size(); ++i)
        shape[i] = policy[i] < 0 ? 0 : mass > 0 ? 0.5 * (1 / count + shape[i] / mass) : 1 / count;
    return shape;
}

double exploration_scale(const Node& node, const SearchConfig& c, double virtual_loss) {
    return (c.pb_c_init + std::log((node.visits + c.pb_c_base + 1) / c.pb_c_base)) *
           std::sqrt(node.visits + node.in_flight * virtual_loss);
}

Node* select_child(Node& node, const SearchConfig& c, bool cheap, double virtual_loss) {
    double mass = 0;
    if (c.use_fpu)
        for (const auto& child : node.children) if (child->visits) mass += child->prior;
    mass = std::min(1.0, mass);
    double mix = std::pow(mass, c.fpu_power);
    double reduction = !node.parent && !cheap ? c.root_fpu : c.fpu;
    double fpu = mix * node.mean() + (1 - mix) * node.nn_value - reduction * std::sqrt(mass);
    double pb_c = exploration_scale(node, c, virtual_loss);
    double child_visits = 0;
    for (const auto& child : node.children) child_visits += child->visits;
    Node* best = nullptr;
    double best_score = -std::numeric_limits<double>::infinity();
    for (const auto& child : node.children) {
        if (child->expanding) continue;
        double value_score = child->visits ? (1 - child->mean()) / 2 : c.use_fpu ? (fpu + 1) / 2 : 0;
        double pending = child->in_flight * virtual_loss;
        if (pending > 0) {
            double weight = std::max(0.25, static_cast<double>(child->visits));
            value_score *= weight / (weight + pending);
        }
        double score = value_score + pb_c * child->prior / (child->visits + pending + 1);
        if (!node.parent && !cheap && child->visits > 0 && child->prior > 0 &&
            child->visits + pending < std::sqrt(child->prior * child_visits * c.root_desired_visits))
            score = 1e20;
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

SearchResult Search::run(const Game& game, Budget budget, bool training, SearchExecution execution) {
    if (game.finished() || budget.visits < 0 || (training && budget.visits == 0) || execution.threads < 1 ||
        !std::isfinite(execution.virtual_loss) || execution.virtual_loss <= 0)
        throw std::runtime_error("Invalid search root, budget or execution settings");
    Node root;
    std::vector<double> network_policy;
    std::array<double, 3> network_wdl;
    const auto& c = config_;
    auto expand = [&](Node* node, Evaluation evaluation) {
        if (evaluation.logits.size() != static_cast<size_t>(game.actions()))
            throw std::runtime_error("Invalid evaluator policy shape");
        for (double probability : evaluation.wdl)
            if (!std::isfinite(probability) || probability < 0 || probability > 1)
                throw std::runtime_error("Invalid evaluator WDL");
        if (std::abs(std::accumulate(evaluation.wdl.begin(), evaluation.wdl.end(), 0.0) - 1) > 1e-5)
            throw std::runtime_error("Unnormalized evaluator WDL");
        for (double logit : evaluation.logits)
            if (!std::isfinite(logit)) throw std::runtime_error("Nonfinite network logit");
        node->hidden = std::move(evaluation.hidden);
        node->nn_value = evaluation.value();
        bool noisy = node == &root && training && !budget.cheap;
        std::vector<int> actions;
        for (int action = 0; action < game.actions(); ++action) {
            if (!game.on_board(action) || (node == &root && !game.legal(action)))
                evaluation.logits[action] = -std::numeric_limits<double>::infinity();
            else actions.push_back(action);
        }
        if (node == &root) {
            network_policy = softmax(evaluation.logits);
            network_wdl = evaluation.wdl;
        }
        if (node == &root && !budget.cheap)
            for (auto& logit : evaluation.logits)
                logit /= c.temperature(c.root_early, c.root_late, game.turn(), game.size());
        auto policy = softmax(evaluation.logits);
        if (noisy && c.noise_weight > 0 && actions.size() > 1) {
            std::vector<double> proportions(actions.size(), 1.0 / actions.size());
            if (c.shaped_noise) {
                std::vector<double> legal_policy;
                for (int action : actions) legal_policy.push_back(policy[action]);
                proportions = dirichlet_alpha_distribution(legal_policy);
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
        node->update(evaluation.wdl);
    };
    expand(&root, evaluator_.initial(game.observation()));
    std::mutex mutex;
    std::condition_variable changed;
    int issued = 0;
    std::exception_ptr failure;
    auto cancelled = [&] { return execution.cancelled && execution.cancelled->load(); };
    auto worker = [&] {
        for (;;) {
            Node* leaf;
            {
                std::unique_lock<std::mutex> lock(mutex);
                for (;;) {
                    if (failure || cancelled() || issued >= budget.visits) return;
                    leaf = &root;
                    while (leaf && !leaf->children.empty())
                        leaf = select_child(*leaf, c, budget.cheap, execution.virtual_loss);
                    if (leaf) break;
                    changed.wait_for(lock, std::chrono::milliseconds(5));
                }
                leaf->expanding = true;
                for (Node* node = leaf; node; node = node->parent) ++node->in_flight;
                ++issued;
            }
            try {
                auto evaluation = evaluator_.recurrent(leaf->parent->hidden, leaf->action);
                std::lock_guard<std::mutex> lock(mutex);
                auto wdl = evaluation.wdl;
                expand(leaf, std::move(evaluation));
                for (Node* node = leaf->parent; node; node = node->parent) {
                    std::swap(wdl[0], wdl[2]);
                    node->update(wdl);
                }
                for (Node* node = leaf; node; node = node->parent) --node->in_flight;
                leaf->expanding = false;
            } catch (...) {
                std::lock_guard<std::mutex> lock(mutex);
                if (!failure) failure = std::current_exception();
                for (Node* node = leaf; node; node = node->parent) --node->in_flight;
                leaf->expanding = false;
            }
            changed.notify_all();
        }
    };
    std::vector<std::thread> workers;
    try {
        int count = std::min(execution.threads, budget.visits);
        for (int i = 1; i < count; ++i) workers.emplace_back(worker);
        worker();
    } catch (...) {
        std::lock_guard<std::mutex> lock(mutex);
        failure = std::current_exception();
    }
    changed.notify_all();
    for (auto& thread : workers) thread.join();
    if (failure) std::rethrow_exception(failure);
    if (root.in_flight != 0) throw std::runtime_error("Unreleased virtual losses");
    auto result = search_result(root, c, game.actions());
    result.network_policy = std::move(network_policy);
    result.network_wdl = network_wdl;
    for (size_t i = 0; i < root.wdl_sum.size(); ++i) result.search_wdl[i] = root.wdl_sum[i] / root.visits;
    for (const auto& child : root.children) {
        double p = result.policy_target[child->action];
        if (p > 0) result.policy_surprise += p * std::log(p / std::max(1e-100, child->prior));
    }
    result.policy_surprise = std::max(0.0, result.policy_surprise);
    result.cancelled = cancelled();
    return result;
}

SearchResult search_result(const Node& root, const SearchConfig& c, int actions) {
    SearchResult result;
    for (const auto& child : root.children) result.completed_visits += child->visits;
    result.visit_policy.resize(actions);
    result.visits.resize(actions);
    for (const auto& child : root.children) {
        result.visits[child->action] = child->visits;
        result.visit_policy[child->action] = result.completed_visits ? child->visits : child->prior;
    }
    result.selection_policy = result.visit_policy;
    const Node* stable = root.children.front().get();
    auto stability = [](const Node& n) { return std::max(0, n.visits - 1) + 2 * n.prior; };
    for (const auto& child : root.children)
        if (child->visits && (!stable->visits || stability(*child) > stability(*stable))) stable = child.get();
    if (c.policy_target_pruning && result.completed_visits) {
        double explore = exploration_scale(root, c, 0);
        double best = (1 - stable->mean()) / 2 + explore * stable->prior / (stable->visits + 1);
        for (const auto& child : root.children) {
            if (child.get() == stable || !child->visits) continue;
            double gap = best - (1 - child->mean()) / 2;
            if (gap > 0) {
                double desired = std::max(0.0, explore * child->prior / gap - 1);
                result.selection_policy[child->action] = std::ceil(std::min(double(child->visits), desired));
            }
        }
    }
    result.move_policy = result.selection_policy;
    if (c.use_lcb && result.completed_visits) {
        std::vector<double> radius(root.children.size(), 2 * c.lcb_stdevs), lcb(root.children.size(), -2 * c.lcb_stdevs);
        for (size_t i = 0; i < root.children.size(); ++i) {
            const auto& child = *root.children[i];
            if (!child.visits) continue;
            double n = child.visits, mean = child.mean(), prior_weight = 1 / (n * n);
            double variance = std::max(1e-8, child.value_sq_sum / n - mean * mean) + prior_weight / (n + prior_weight);
            double effective = std::pow(n + prior_weight, 2) / (n + prior_weight * prior_weight);
            radius[i] = c.lcb_stdevs * std::sqrt(variance / effective);
            lcb[i] = -mean - radius[i];
        }
        int best = -1;
        for (size_t i = 0; i < root.children.size(); ++i) {
            const auto& child = *root.children[i];
            if (result.selection_policy[child.action] > 0 &&
                result.selection_policy[child.action] >= c.min_lcb_visits * stable->visits &&
                (best < 0 || lcb[i] > lcb[best])) best = static_cast<int>(i);
        }
        if (best < 0) throw std::runtime_error("No LCB candidate");
        int action = root.children[best]->action;
        for (size_t i = 0; i < root.children.size(); ++i) {
            const auto& child = *root.children[i];
            double excess = lcb[best] - lcb[i];
            if (static_cast<int>(i) == best || !child.visits || excess <= 0) continue;
            double factor = (radius[i] + excess) / (radius[i] + 0.20 * excess);
            result.selection_policy[action] = std::max(result.selection_policy[action], factor * factor * result.selection_policy[child.action]);
        }
    }
    for (auto* policy : {&result.visit_policy, &result.move_policy, &result.selection_policy}) {
        if (c.policy_target_pruning && policy != &result.visit_policy) {
            double threshold = std::min(1.0, *std::max_element(policy->begin(), policy->end()) / 64);
            for (auto& weight : *policy) if (weight < threshold) weight = 0;
        }
        double mass = std::accumulate(policy->begin(), policy->end(), 0.0);
        for (auto& weight : *policy) weight /= mass;
    }
    result.policy_target = result.selection_policy;
    result.root_value = root.mean();
    return result;
}

}
