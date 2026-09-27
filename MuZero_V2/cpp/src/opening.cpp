#include "muzero/opening.h"
#include <limits>

namespace muzero {

OpeningConfig::OpeningConfig(const Config& c)
    : probability(c.number("BALANCED_OPENING_PROB")), avg_dist_factor(c.number("BALANCED_OPENING_AVG_DIST_FACTOR")),
      balance_exponent(c.number("BALANCED_OPENING_BALANCE_EXPONENT")),
      rejection_probability(c.number("BALANCED_OPENING_REJECTION_PROB")),
      rejection_probability_fallback(c.number("BALANCED_OPENING_REJECTION_PROB_FALLBACK")),
      max_tries(c.integer("BALANCED_OPENING_MAX_TRIES")), policy_init(c.boolean("INIT_GAMES_WITH_POLICY")),
      policy_after(c.boolean("BALANCED_OPENING_RUN_POLICY_INIT_AFTER")),
      policy_on_failure(c.boolean("BALANCED_OPENING_RUN_POLICY_INIT_ON_FAILURE")),
      policy_init_mean(c.number("POLICY_INIT_AVG_MOVE_NUM")), policy_temperature(c.number("POLICY_INIT_TEMPERATURE")) {
    for (double p : {probability, rejection_probability, rejection_probability_fallback})
        if (p < 0 || p > 1) throw std::runtime_error("Invalid opening probability");
    if (max_tries < 1 || max_tries > 1000 || avg_dist_factor < 0 || avg_dist_factor > 100 ||
        balance_exponent < 0 || balance_exponent > 100 || policy_init_mean < 0 || policy_init_mean > 100 ||
        policy_temperature < 0.1 || policy_temperature > 5)
        throw std::runtime_error("Invalid opening configuration");
}

namespace {
class OpeningRandom {
    std::mt19937_64& engine_;
    bool has_gaussian_ = false;
    double gaussian_ = 0;
public:
    explicit OpeningRandom(std::mt19937_64& engine) : engine_(engine) {}
    double uniform() { return static_cast<double>(engine_() & ((1ULL << 53) - 1)) / static_cast<double>(1ULL << 53); }
    bool coin(double p) { return uniform() < p; }
    int uniform_index(int count) { return std::uniform_int_distribution<int>(0, count - 1)(engine_); }
    double exponential() {
        double r = 0;
        while (r <= 1e-17) r = uniform();
        return -std::log(r);
    }
    double gaussian() {
        if (has_gaussian_) { has_gaussian_ = false; return gaussian_; }
        double v1, v2, s;
        do {
            v1 = uniform() * 2 - 1;
            v2 = uniform() * 2 - 1;
            s = v1 * v1 + v2 * v2;
        } while (s >= 1 || s == 0);
        double multiplier = std::sqrt(-2 * std::log(s) / s);
        gaussian_ = v2 * multiplier;
        has_gaussian_ = true;
        return v1 * multiplier;
    }
    double truncated_gaussian(double bound) {
        double d = gaussian();
        while (d < -bound || d > bound) d = gaussian();
        return d;
    }
    int choose(const std::vector<double>& weights) {
        double r = uniform() * std::accumulate(weights.begin(), weights.end(), 0.0);
        double sum = 0;
        for (size_t i = 0; i < weights.size(); ++i) {
            sum += weights[i];
            if (sum > r) return static_cast<int>(i);
        }
        return static_cast<int>(weights.size() - 1);
    }
};

struct Cancelled {};

class Initializer {
    const OpeningConfig& config_;
    Evaluator& evaluator_;
    OpeningRandom random_;
    const std::function<bool()>& cancelled_;
    OpeningResult result_;

    void check_cancelled() const {
        if (cancelled_ && cancelled_()) throw Cancelled{};
    }
    double value(const Game& game, int player) {
        check_cancelled();
        return evaluator_.initial(game.observation(player)).value();
    }
    int nearby_move(const Game& game, double avg_dist) {
        int size = game.size();
        if (game.turn() == 0) {
            double xd = random_.truncated_gaussian(1.5 * 0.999) / 3.0;
            double yd = random_.truncated_gaussian(1.5 * 0.999) / 3.0;
            int x = static_cast<int>(std::round(xd * size + 0.5 * (size - 1)));
            int y = static_cast<int>(std::round(yd * size + 0.5 * (size - 1)));
            if (x < 0 || x >= size || y < 0 || y >= size) return -1;
            return y * game.canvas() + x;
        }
        std::vector<double> weights(size * size, 0.0);
        for (int x1 = 0; x1 < size; ++x1) for (int y1 = 0; y1 < size; ++y1) {
            if (game.board().cells[y1 * size + x1] == 0) continue;
            for (int x2 = 0; x2 < size; ++x2) for (int y2 = 0; y2 < size; ++y2) {
                if (game.board().cells[y2 * size + x2] != 0) continue;
                double half = 0.5 * (size - 1);
                double distance = std::max(std::abs(x2 - half), std::abs(y2 - half));
                double bonus = 1.5 * (half - distance) / half;
                double dx = x2 - x1, dy = y2 - y1;
                weights[y2 * size + x2] += (1 + bonus) * std::pow(dx * dx + dy * dy + avg_dist * avg_dist, -2.0);
            }
        }
        double total = std::accumulate(weights.begin(), weights.end(), 0.0);
        if (!(total > 0) || !std::isfinite(total)) return -1;
        int local = random_.choose(weights);
        return local / size * game.canvas() + local % size;
    }
    int balance_move(const Game& game, double rejection) {
        double root = value(game, game.player());
        if (!std::isfinite(root)) { result_.failure = "nonfinite root evaluation"; return -1; }
        if (root < 0 && random_.coin(1 - std::exp(-3 * root * root)) && random_.coin(rejection)) return -1;
        double opposite = value(game, -game.player());
        if (!std::isfinite(opposite)) { result_.failure = "nonfinite counterfactual root evaluation"; return -1; }
        if (opposite < 0 && random_.coin(1 - std::exp(-3 * opposite * opposite)) && random_.coin(rejection)) return -1;
        bool nearby = opposite > 0 && game.turn() > 0;
        int size = game.size();
        std::vector<double> weights(size * size, 0), values(size * size, 0);
        double max_probability = 0;
        for (int x = 0; x < size; ++x) for (int y = 0; y < size; ++y) {
            int action = y * game.canvas() + x;
            if (!game.legal(action)) continue;
            if (nearby) {
                bool near_stone = false;
                for (int dx = -3; dx <= 3 && !near_stone; ++dx)
                    for (int dy = -3; dy <= 3 && !near_stone; ++dy) {
                        int xx = x + dx, yy = y + dy;
                        if (xx >= 0 && xx < size && yy >= 0 && yy < size)
                            near_stone = game.board().cells[yy * size + xx] != 0;
                    }
                if (!near_stone) continue;
            }
            Game candidate = game;
            candidate.play(action);
            if (candidate.finished()) continue;
            double v = value(candidate, candidate.player());
            if (!std::isfinite(v)) { result_.failure = "nonfinite candidate evaluation"; return -1; }
            double p = std::pow(std::max(0.0, 1 - v * v), config_.balance_exponent);
            weights[y * size + x] = p;
            values[y * size + x] = v;
            max_probability = std::max(max_probability, p);
        }
        if (random_.coin(1 - max_probability) && random_.coin(rejection)) return -1;
        double total = std::accumulate(weights.begin(), weights.end(), 0.0);
        if (!(total > 0) || !std::isfinite(total)) return -1;
        int local = random_.choose(weights);
        result_.start_value = values[local];
        return local / size * game.canvas() + local % size;
    }
    void balanced(Game& game) {
        result_.status = OpeningStatus::Failed;
        if (game.finished() || game.turn() != 0) {
            result_.failure = "opening requires an empty ongoing position";
            return;
        }
        double rejection = config_.rejection_probability;
        int tries_at_probability = 0;
        while (true) {
            check_cancelled();
            ++result_.attempts;
            Game candidate = game;
            std::vector<int> actions;
            int count = random_.choose({10, 30, 50, 80, 60, 40, 20, 10, 5, 1, 0, 0});
            double distance = random_.exponential() * config_.avg_dist_factor;
            bool valid = true;
            for (int i = 0; i < count; ++i) {
                int action = nearby_move(candidate, distance);
                if (action < 0 || !candidate.legal(action)) { valid = false; break; }
                candidate.play(action);
                if (candidate.finished()) { valid = false; break; }
                actions.push_back(action);
            }
            if (valid) {
                random_.coin(0.5);
                int action = balance_move(candidate, rejection);
                if (!result_.failure.empty()) return;
                if (action >= 0) {
                    candidate.play(action);
                    if (!candidate.finished()) {
                        actions.push_back(action);
                        game = std::move(candidate);
                        result_.actions = std::move(actions);
                        result_.balanced_moves = game.turn();
                        result_.status = OpeningStatus::Success;
                        return;
                    }
                }
            }
            if (++tries_at_probability > config_.max_tries) {
                tries_at_probability = 0;
                rejection = config_.rejection_probability_fallback;
            }
        }
    }
    void policy(Game& game) {
        if (game.finished() || config_.policy_init_mean <= 0) return;
        int count = std::max(0, static_cast<int>(std::floor(random_.exponential() * config_.policy_init_mean - 2.0 * game.turn())));
        for (int i = 0; i < count; ++i) {
            check_cancelled();
            auto evaluation = evaluator_.initial(game.observation());
            if (evaluation.logits.size() != static_cast<size_t>(game.actions()))
                throw std::runtime_error("Invalid opening policy shape");
            for (int a = 0; a < game.actions(); ++a) {
                if (!std::isfinite(evaluation.logits[a])) throw std::runtime_error("Nonfinite opening policy logit");
                if (!game.legal(a)) evaluation.logits[a] = -std::numeric_limits<double>::infinity();
            }
            auto probabilities = softmax(evaluation.logits);
            std::vector<int> actions;
            std::vector<double> weights;
            for (int a = 0; a < game.actions(); ++a) if (game.legal(a) && probabilities[a] > 0) {
                actions.push_back(a);
                weights.push_back(std::pow(probabilities[a], 1 / config_.policy_temperature));
            }
            if (actions.empty()) throw std::runtime_error("Empty opening policy");
            int chosen = random_.coin(0.0002) ? random_.uniform_index(static_cast<int>(actions.size())) : random_.choose(weights);
            int action = actions[chosen];
            game.play(action);
            result_.actions.push_back(action);
            ++result_.policy_moves;
            if (game.finished()) break;
        }
    }
public:
    Initializer(const OpeningConfig& config, Evaluator& evaluator, std::mt19937_64& random,
                const std::function<bool()>& cancelled)
        : config_(config), evaluator_(evaluator), random_(random), cancelled_(cancelled) {}
    OpeningResult run(Game& game) {
        try {
            check_cancelled();
            if (!game.finished() && config_.probability > 0 && random_.coin(config_.probability)) balanced(game);
            bool run_policy = result_.status == OpeningStatus::NotAttempted ||
                (result_.status == OpeningStatus::Success ? config_.policy_after : config_.policy_on_failure);
            if (run_policy && config_.policy_init) policy(game);
        } catch (const Cancelled&) {
            result_.status = OpeningStatus::Interrupted;
        }
        return std::move(result_);
    }
};
}

OpeningResult initialize_opening(Game& game, const OpeningConfig& config, Evaluator& evaluator,
                                 std::mt19937_64& random, const std::function<bool()>& cancelled) {
    return Initializer(config, evaluator, random, cancelled).run(game);
}

}
