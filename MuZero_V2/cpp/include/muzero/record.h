#pragma once

#include "search.h"
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace muzero {

struct Step {
    int player, action;
    Budget budget;
    std::vector<float> observation;
    std::vector<double> policy;
};

inline std::filesystem::path game_path(const std::filesystem::path& directory, int index) {
    std::ostringstream name;
    name << "game_" << std::setw(8) << std::setfill('0') << index << ".mzg";
    return directory / name.str();
}

class RecordWriter {
    std::ofstream output_;
public:
    explicit RecordWriter(const std::filesystem::path& path) : output_(path, std::ios::binary | std::ios::trunc) {
        output_.exceptions(std::ios::badbit | std::ios::failbit);
    }
    void integer(uint32_t value) {
        char bytes[4];
        for (int i = 0; i < 4; ++i) bytes[i] = static_cast<char>((value >> (8 * i)) & 255);
        output_.write(bytes, 4);
    }
    void floating(float value) {
        static_assert(sizeof(float) == sizeof(uint32_t));
        uint32_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        integer(bits);
    }
    void game(const Game& game, const std::vector<Step>& steps) {
        output_.write(GAME_MAGIC, 8);
        integer(game.canvas());
        integer(game.size());
        integer(static_cast<uint32_t>(game.rule()));
        integer(static_cast<uint32_t>(game.winner()));
        integer(static_cast<uint32_t>(steps.size()));
        for (const auto& step : steps) {
            integer(static_cast<uint32_t>(step.player));
            integer(step.action);
            floating(step.budget.cheap ? 0.0f : 1.0f);
            integer(step.budget.visits);
            for (float value : step.observation) output_.put(static_cast<char>(value));
            for (double value : step.policy) floating(static_cast<float>(value));
        }
        output_.flush();
        output_.close();
    }
};

}
