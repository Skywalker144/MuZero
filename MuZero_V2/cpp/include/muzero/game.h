#pragma once

#include "rules.h"
#include <stdexcept>

namespace muzero {

class Game {
    Board board_;
    int canvas_, player_ = 1, turn_ = 0, winner_ = 0;
    Rule rule_;
    bool finished_ = false;
public:
    explicit Game(int size, int canvas = 0, Rule rule = Rule::FREESTYLE)
        : board_(size), canvas_(canvas ? canvas : size), rule_(rule) {
        if (size < 5 || size > 25 || canvas_ < size || canvas_ > 25)
            throw std::runtime_error("Invalid board dimensions");
    }
    int size() const { return board_.size; }
    int canvas() const { return canvas_; }
    int actions() const { return canvas_ * canvas_; }
    Rule rule() const { return rule_; }
    int player() const { return player_; }
    int turn() const { return turn_; }
    int winner() const { return winner_; }
    bool finished() const { return finished_; }
    const Board& board() const { return board_; }
    bool on_board(int action) const {
        return action >= 0 && action < actions() && action / canvas_ < size() && action % canvas_ < size();
    }
    bool legal(int action) const {
        return !finished_ && on_board(action) && board_.cells[action / canvas_ * size() + action % canvas_] == 0;
    }
    std::vector<float> observation() const { return observation(player_); }
    std::vector<float> observation(int player) const {
        if (player != 1 && player != -1) throw std::runtime_error("Invalid observation player");
        std::vector<float> result(INPUT_PLANES * actions());
        RenjuAnalyzer analyzer;
        for (int y = 0; y < size(); ++y) {
            for (int x = 0; x < size(); ++x) {
                int local = y * size() + x, action = y * canvas_ + x;
                auto set = [&](Plane plane, bool value) { result[static_cast<int>(plane) * actions() + action] = value; };
                set(Plane::OWN, board_.cells[local] == player);
                set(Plane::OPPONENT, board_.cells[local] == -player);
                set(Plane::BLACK_TO_MOVE, player == 1);
                set(Plane::ON_BOARD, true);
                set(Plane::STANDARD, rule_ == Rule::STANDARD);
                set(Plane::RENJU, rule_ == Rule::RENJU);
                if (rule_ == Rule::RENJU)
                    set(player == 1 ? Plane::FORBIDDEN_BLACK_TURN : Plane::FORBIDDEN_WHITE_TURN,
                        analyzer.forbidden(board_, local));
            }
        }
        return result;
    }
    void play(int action) {
        if (!legal(action)) throw std::runtime_error("Illegal move");
        int local = action / canvas_ * size() + action % canvas_;
        RenjuAnalyzer analyzer;
        if (rule_ == Rule::RENJU && player_ == 1 && analyzer.forbidden(board_, local)) winner_ = -1;
        board_.cells[local] = player_;
        ++turn_;
        if (!winner_) {
            for (int length : board_.lengths(local, player_))
                if (length == 5 || (length > 5 && (rule_ == Rule::FREESTYLE || (rule_ == Rule::RENJU && player_ == -1))))
                    winner_ = player_;
        }
        finished_ = winner_ != 0 || turn_ == size() * size();
        player_ = -player_;
    }
};

}
