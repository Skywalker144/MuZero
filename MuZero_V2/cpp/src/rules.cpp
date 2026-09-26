#include "muzero/rules.h"
#include <set>
#include <stdexcept>

namespace muzero {

int Board::offset(int origin, int direction, int distance) const {
    constexpr int dx[] = {1, 0, 1, 1}, dy[] = {0, 1, 1, -1};
    int x = origin % size + dx[direction] * distance;
    int y = origin / size + dy[direction] * distance;
    return x >= 0 && x < size && y >= 0 && y < size ? y * size + x : -1;
}

std::array<int, 4> Board::lengths(int action, int player) const {
    std::array<int, 4> result{};
    for (int d = 0; d < 4; ++d) {
        result[d] = 1;
        for (int sign : {-1, 1}) {
            for (int distance = sign; ; distance += sign) {
                int loc = offset(action, d, distance);
                if (loc < 0 || cells[loc] != player) break;
                ++result[d];
            }
        }
    }
    return result;
}

Rule parse_rule(const std::string& name) {
    for (int i = 0; i < static_cast<int>(std::size(RULE_NAMES)); ++i)
        if (name == RULE_NAMES[i]) return static_cast<Rule>(i);
    throw std::runtime_error("Unknown Gomoku rule: " + name);
}

namespace {
std::vector<int> run(const Board& board, int anchor, int direction) {
    std::vector<int> negative;
    for (int distance = -1; ; --distance) {
        int loc = board.offset(anchor, direction, distance);
        if (loc < 0 || board.cells[loc] != 1) break;
        negative.push_back(loc);
    }
    std::reverse(negative.begin(), negative.end());
    negative.push_back(anchor);
    for (int distance = 1; ; ++distance) {
        int loc = board.offset(anchor, direction, distance);
        if (loc < 0 || board.cells[loc] != 1) break;
        negative.push_back(loc);
    }
    return negative;
}
}

Forbidden RenjuAnalyzer::analyze(const Board& board, int action) {
    if (action < 0 || action >= board.size * board.size || board.cells[action] != 0)
        return Forbidden::NONE;
    int nearby = 0;
    for (int d = 0; d < 4; ++d)
        for (int distance : {-2, -1, 1, 2}) {
            int loc = board.offset(action, d, distance);
            nearby += loc >= 0 && board.cells[loc] == 1;
        }
    if (nearby < 2) return Forbidden::NONE;
    auto key = std::make_pair(board.cells, action);
    auto found = cache_.find(key);
    if (found != cache_.end()) return found->second;
    Board after = board;
    after.cells[action] = 1;
    auto lengths = after.lengths(action, 1);
    if (std::find(lengths.begin(), lengths.end(), 5) != lengths.end()) return cache_[key] = Forbidden::NONE;
    if (*std::max_element(lengths.begin(), lengths.end()) > 5) return cache_[key] = Forbidden::OVERLINE;
    std::set<std::vector<int>> fours, threes;
    for (int target : {5, 4}) {
        for (int d = 0; d < 4; ++d) {
            for (int distance = -4; distance <= 4; ++distance) {
                int extension = after.offset(action, d, distance);
                if (distance == 0 || extension < 0 || after.cells[extension] != 0) continue;
                Board extended = after;
                extended.cells[extension] = 1;
                auto stones = run(extended, extension, d);
                if (static_cast<int>(stones.size()) != target ||
                    std::find(stones.begin(), stones.end(), action) == stones.end()) continue;
                if (target == 4) {
                    auto extension_lengths = extended.lengths(extension, 1);
                    if (std::find(extension_lengths.begin(), extension_lengths.end(), 5) != extension_lengths.end()) continue;
                    int left = extended.offset(stones.front(), d, -1), right = extended.offset(stones.back(), d, 1);
                    if (left < 0 || right < 0 || extended.cells[left] != 0 || extended.cells[right] != 0 ||
                        extended.lengths(left, 1)[d] != 5 || extended.lengths(right, 1)[d] != 5 || forbidden(after, extension)) continue;
                }
                stones.erase(std::find(stones.begin(), stones.end(), extension));
                std::sort(stones.begin(), stones.end());
                (target == 5 ? fours : threes).insert(stones);
            }
        }
        if (target == 5 && fours.size() >= 2) return cache_[key] = Forbidden::DOUBLE_FOUR;
    }
    return cache_[key] = threes.size() >= 2 ? Forbidden::DOUBLE_THREE : Forbidden::NONE;
}

}
