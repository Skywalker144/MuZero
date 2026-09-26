#pragma once

#include "muzero/protocol.h"
#include <algorithm>
#include <array>
#include <map>
#include <string>
#include <stdexcept>
#include <vector>

namespace muzero {

struct Board {
    int size;
    std::vector<int> cells;
    explicit Board(int edge) : size(edge) {
        if (edge < 5 || edge > 25) throw std::runtime_error("Invalid board size");
        cells.assign(edge * edge, 0);
    }
    int offset(int origin, int direction, int distance) const;
    std::array<int, 4> lengths(int action, int player) const;
};

enum class Forbidden { NONE, OVERLINE, DOUBLE_FOUR, DOUBLE_THREE };

class RenjuAnalyzer {
    std::map<std::pair<std::vector<int>, int>, Forbidden> cache_;
public:
    Forbidden analyze(const Board& board, int action);
    bool forbidden(const Board& board, int action) { return analyze(board, action) != Forbidden::NONE; }
};

Rule parse_rule(const std::string& name);

}
