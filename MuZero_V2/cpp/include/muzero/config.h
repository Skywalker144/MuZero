#pragma once

#include <cmath>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>

namespace muzero {

class Config {
    std::map<std::string, std::string> values_;
public:
    explicit Config(const std::string& path) {
        std::ifstream input(path);
        if (!input) throw std::runtime_error("Cannot read config: " + path);
        std::string line;
        while (std::getline(input, line)) {
            if (line.empty()) continue;
            auto pos = line.find('=');
            if (pos == std::string::npos) throw std::runtime_error("Expected resolved KEY=value config");
            auto key = line.substr(0, pos);
            if (!values_.emplace(key, line.substr(pos + 1)).second)
                throw std::runtime_error("Duplicate config: " + key);
        }
    }
    const std::string& text(const std::string& key) const {
        auto found = values_.find(key);
        if (found == values_.end()) throw std::runtime_error("Missing config: " + key);
        return found->second;
    }
    double number(const std::string& key) const {
        size_t consumed = 0;
        const auto& value = text(key);
        double result = std::stod(value, &consumed);
        if (consumed != value.size() || !std::isfinite(result))
            throw std::runtime_error("Invalid number: " + key);
        return result;
    }
    int integer(const std::string& key) const {
        size_t consumed = 0;
        const auto& value = text(key);
        int result = std::stoi(value, &consumed);
        if (consumed != value.size()) throw std::runtime_error("Invalid integer: " + key);
        return result;
    }
    bool boolean(const std::string& key) const {
        auto value = integer(key);
        if (value != 0 && value != 1) throw std::runtime_error("Invalid boolean: " + key);
        return value == 1;
    }
};

struct SearchConfig {
    int full_search_visits, cheap_visits;
    double cheap_probability, pb_c_init, pb_c_base, concentration, noise_weight;
    bool shaped_noise, use_lcb, use_fpu;
    std::string move_schedule;
    int temperature_moves;
    double lcb_stdevs, min_lcb_visits, move_early, move_late, halflife;
    double root_early, root_late, fpu, root_fpu, fpu_power;
    explicit SearchConfig(const Config& c)
        : full_search_visits(c.integer("FULL_SEARCH_VISITS")), cheap_visits(c.integer("CHEAP_SEARCH_VISITS")),
          cheap_probability(c.number("CHEAP_SEARCH_PROB")), pb_c_init(c.number("PB_C_INIT")),
          pb_c_base(c.number("PB_C_BASE")), concentration(c.number("DIRICHLET_TOTAL_CONCENTRATION")),
          noise_weight(c.number("DIRICHLET_NOISE_WEIGHT")), shaped_noise(c.boolean("SHAPED_DIRICHLET_NOISE")),
          use_lcb(c.boolean("USE_LCB_FOR_SELECTION")), use_fpu(c.boolean("USE_FPU")),
          move_schedule(c.text("MOVE_TEMPERATURE_SCHEDULE")), temperature_moves(c.integer("TEMPERATURE_MOVES")),
          lcb_stdevs(c.number("LCB_STDEVS")),
          min_lcb_visits(c.number("MIN_VISIT_PROP_FOR_LCB")),
          move_early(c.number("CHOSEN_MOVE_TEMPERATURE_EARLY")), move_late(c.number("CHOSEN_MOVE_TEMPERATURE")),
          halflife(c.number("CHOSEN_MOVE_TEMPERATURE_HALFLIFE")),
          root_early(c.number("ROOT_POLICY_TEMPERATURE_EARLY")), root_late(c.number("ROOT_POLICY_TEMPERATURE")),
          fpu(c.number("FPU_REDUCTION_MAX")), root_fpu(c.number("ROOT_FPU_REDUCTION_MAX")),
          fpu_power(c.number("FPU_PARENT_WEIGHT_BY_VISITED_POLICY_POW")) {
        if (full_search_visits < 1 || cheap_visits < 1 || cheap_probability < 0 || cheap_probability > 1 ||
            noise_weight < 0 || noise_weight > 1 || pb_c_base <= 0 || pb_c_init < 0 ||
            concentration <= 0 || lcb_stdevs <= 0 || min_lcb_visits < 0 || min_lcb_visits > 1 ||
            move_early < 0 || move_late < 0 || halflife <= 0 || root_early <= 0 || root_late <= 0 ||
            fpu < 0 || root_fpu < 0 || fpu_power <= 0 || temperature_moves < 0 ||
            (move_schedule != "exponential" && move_schedule != "threshold"))
            throw std::runtime_error("Invalid search configuration");
    }
    double temperature(double early, double late, int turn, int size) const {
        return late + (early - late) * std::pow(0.5, turn * 19.0 / (halflife * size));
    }
    double move_temperature(int turn, int size) const {
        if (move_schedule == "threshold") return turn < temperature_moves ? move_early : move_late;
        return temperature(move_early, move_late, turn, size);
    }
};

}
