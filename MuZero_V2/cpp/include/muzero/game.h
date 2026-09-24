#pragma once

#include <array>
#include <stdexcept>
#include <vector>

namespace muzero {

class Game {
    int size_, player_ = 1, turn_ = 0, winner_ = 0;
    bool finished_ = false;
    std::vector<int> board_;
public:
    explicit Game(int size) : size_(size) {
        if (size < 5 || size > 25)
            throw std::runtime_error("Invalid board dimensions");
        board_.assign(size * size, 0);
    }
    int size() const { return size_; }
    int actions() const { return size_ * size_; }
    int player() const { return player_; }
    int turn() const { return turn_; }
    int winner() const { return winner_; }
    bool finished() const { return finished_; }
    bool legal(int action) const { return !finished_ && action >= 0 && action < actions() && board_[action] == 0; }
    std::vector<float> observation() const {
        std::vector<float> result(3 * actions());
        for (int i = 0; i < actions(); ++i) {
            result[i] = board_[i] == player_;
            result[actions() + i] = board_[i] == -player_;
            result[2 * actions() + i] = player_ == 1;
        }
        return result;
    }
    void play(int action) {
        if (!legal(action)) throw std::runtime_error("Illegal move");
        board_[action] = player_;
        ++turn_;
        const std::array<std::array<int, 2>, 4> directions{{{1, 0}, {0, 1}, {1, 1}, {1, -1}}};
        for (auto direction : directions) {
            int count = 1;
            for (int sign : {-1, 1}) {
                int r = action / size_ + sign * direction[0], c = action % size_ + sign * direction[1];
                while (r >= 0 && r < size_ && c >= 0 && c < size_ && board_[r * size_ + c] == player_) {
                    ++count;
                    r += sign * direction[0];
                    c += sign * direction[1];
                }
            }
            if (count >= 5) {
                winner_ = player_;
                finished_ = true;
            }
        }
        finished_ = finished_ || turn_ == actions();
        player_ = -player_;
    }
};

}
