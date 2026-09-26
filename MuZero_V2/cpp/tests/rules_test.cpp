#include "muzero/game.h"
#include <iostream>

using namespace muzero;

void check(bool value) {
    if (!value) throw std::runtime_error("Rule assertion failed");
}

void forbidden_case(std::initializer_list<std::pair<int, int>> points, int x, int y, bool expected) {
    Board board(15);
    for (auto point : points) board.cells[point.second * 15 + point.first] = 1;
    RenjuAnalyzer analyzer;
    check(analyzer.forbidden(board, y * 15 + x) == expected);
}

int main() {
    try {
        for (Rule rule : {Rule::FREESTYLE, Rule::STANDARD, Rule::RENJU}) {
            Game game(6, 9, rule);
            for (int local : {0, 6, 1, 8, 3, 10, 4, 12, 5, 14, 2})
                game.play(local / 6 * 9 + local % 6);
            check(game.finished() == (rule != Rule::STANDARD));
            check(game.winner() == (rule == Rule::FREESTYLE ? 1 : rule == Rule::RENJU ? -1 : 0));
            check(!game.legal(8));
            check(game.observation().size() == INPUT_PLANES * 81);
        }
        for (Rule rule : {Rule::FREESTYLE, Rule::STANDARD, Rule::RENJU}) {
            Game white(6, 6, rule);
            for (int action : {6, 0, 8, 1, 10, 3, 12, 4, 14, 5, 16, 2}) white.play(action);
            check(white.finished() == (rule != Rule::STANDARD));
            check(white.winner() == (rule == Rule::STANDARD ? 0 : -1));
            Game five(5, 7, rule);
            for (int action : {0, 7, 1, 8, 2, 9, 3, 10, 4}) five.play(action);
            check(five.finished() && five.winner() == 1);
            Game draw(5, 7, rule);
            for (int local : {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 10, 13, 12, 15, 14, 17, 16, 19, 18, 20, 21, 22, 23, 24})
                draw.play(local / 5 * 7 + local % 5);
            check(draw.finished() && draw.winner() == 0);
        }
        {
            Game renju(15, 15, Rule::RENJU);
            for (int action : {111, 0, 113, 2, 97, 4, 127, 6}) renju.play(action);
            auto observation = renju.observation();
            check(renju.legal(112));
            check(observation[static_cast<int>(Plane::FORBIDDEN_BLACK_TURN) * 225 + 112] == 1);
            renju.play(112);
            check(renju.finished() && renju.winner() == -1);
        }
        forbidden_case({{6,7},{8,7},{7,6},{7,8}}, 7, 7, true);
        forbidden_case({{5,7},{6,7},{8,7},{7,5},{7,6},{7,8}}, 7, 7, true);
        forbidden_case({{1,0},{2,0},{0,1},{0,2}}, 0, 0, false);
        forbidden_case({{3,7},{4,7},{5,7},{6,7},{7,6},{7,8},{6,6},{8,8}}, 7, 7, false);
        forbidden_case({{5,7},{6,7},{8,7},{9,7},{7,4},{7,5},{7,6},{7,8},{7,9}}, 7, 7, false);
        forbidden_case({{6,1},{8,1},{6,3},{7,3},{9,3},{10,3},{9,4}}, 7, 2, false);
        std::cout << "rules passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
