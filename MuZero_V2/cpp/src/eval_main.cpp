#include "muzero/torch_backend.h"
#include "muzero/eval_session.h"
#include <ATen/Parallel.h>
#include <charconv>
#include <iomanip>
#include <iostream>
#include <string_view>

using namespace muzero;

int parse_integer(std::string_view text) {
    int value;
    auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size())
        throw std::runtime_error("Expected integer argument");
    return value;
}

void write_array(const std::vector<int>& values) {
    std::cout << '[';
    for (size_t i = 0; i < values.size(); ++i) {
        if (i) std::cout << ',';
        std::cout << values[i];
    }
    std::cout << ']';
}

void write_state(const EvalSession& session) {
    const auto& game = session.game();
    std::cout << "{\"board_size\":" << game.size() << ",\"canvas_size\":" << game.canvas()
              << ",\"player\":" << game.player() << ",\"turn\":" << game.turn()
              << ",\"finished\":" << (game.finished() ? "true" : "false") << ",\"winner\":" << game.winner()
              << ",\"board\":";
    write_array(game.board().cells);
    std::cout << ",\"moves\":";
    write_array(session.moves());
    std::cout << '}';
}

void write_analysis(const Analysis& analysis, const Game& game, uint64_t requests, uint64_t batches) {
    const auto& result = analysis.result;
    std::vector<int> candidates;
    for (int i = 0; i < game.actions(); ++i) if (game.legal(i)) candidates.push_back(i);
    std::stable_sort(candidates.begin(), candidates.end(), [&](int a, int b) {
        return result.selection_policy[a] > result.selection_policy[b];
    });
    std::cout << "{\"action\":" << analysis.action
              << ",\"board_size\":" << game.size() << ",\"canvas_size\":" << game.canvas()
              << ",\"turn\":" << analysis.turn << ",\"player\":" << analysis.player
              << ",\"root_value\":" << result.root_value
              << ",\"completed_visits\":" << result.completed_visits << ",\"seconds\":" << analysis.seconds
              << ",\"requests\":" << requests << ",\"batches\":" << batches << ",\"board\":";
    write_array(game.board().cells);
    std::cout << ",\"candidates\":[";
    for (size_t i = 0; i < candidates.size(); ++i) {
        int candidate = candidates[i];
        if (i) std::cout << ',';
        std::cout << "{\"action\":" << candidate / game.canvas() * game.size() + candidate % game.canvas()
                  << ",\"visits\":" << result.visits[candidate]
                  << ",\"network_prior\":" << result.network_policy[candidate]
                  << ",\"visit_policy\":" << result.visit_policy[candidate]
                  << ",\"selection_weight\":" << result.selection_policy[candidate] << '}';
    }
    std::cout << "]}";
}

void write_error(const std::string& error) {
    std::cout << "{\"ok\":false,\"error\":\"";
    for (unsigned char c : error) {
        if (c == '"' || c == '\\') std::cout << '\\' << c;
        else if (c < 32) std::cout << ' ';
        else std::cout << c;
    }
    std::cout << "\"}\n" << std::flush;
}

int main(int argc, char** argv) {
    try {
        bool serve = argc == 4 && std::string(argv[3]) == "--serve";
        if (!serve && argc < 5) throw std::runtime_error("Usage: muzero_eval resolved.cfg model.pt (--serve | board_size rule [row_major_moves...])");
        Config raw(argv[1]);
        SearchConfig config(raw, SearchProfile::Evaluation);
        at::set_num_threads(raw.integer("TORCH_THREADS"));
        auto backend = std::make_unique<TorchBackend>(argv[2], raw.text("DEVICE"), 0, raw.number("NN_POLICY_TEMPERATURE"));
        int canvas = backend->canvas_size();
        BatchEvaluator evaluator(std::move(backend), raw.integer("NN_MAX_BATCH_SIZE"), raw.integer("NN_BATCH_WAIT_US"));
        EvalSession session(canvas, config, evaluator,
                            {raw.integer("NUM_SEARCH_THREADS"), raw.number("VIRTUAL_LOSS")}, raw.integer("SEED"));
        std::cout << std::setprecision(17);
        if (!serve) {
            session.reset(parse_integer(argv[3]), parse_rule(argv[4]));
            for (int i = 5; i < argc; ++i) session.play(parse_integer(argv[i]));
            auto analysis = session.analyze(config.full_search_visits);
            write_analysis(analysis, session.game(), evaluator.requests.load(), evaluator.batches.load());
            std::cout << '\n';
            return 0;
        }
        std::cout << "{\"ok\":true,\"canvas_size\":" << canvas << "}\n" << std::flush;
        std::string line;
        while (std::getline(std::cin, line)) {
            try {
                std::istringstream input(line);
                std::vector<std::string> words;
                for (std::string word; input >> word;) words.push_back(word);
                if (words.empty()) throw std::runtime_error("Empty command");
                if (words[0] == "quit" && words.size() == 1) break;
                std::optional<Analysis> analysis;
                std::optional<Game> analyzed_game;
                uint64_t requests = evaluator.requests.load(), batches = evaluator.batches.load();
                if (words[0] == "new" && words.size() == 3)
                    session.reset(parse_integer(words[1]), parse_rule(words[2]));
                else if (words[0] == "play" && words.size() == 2) session.play(parse_integer(words[1]));
                else if (words[0] == "undo" && words.size() == 2) session.undo(parse_integer(words[1]));
                else if ((words[0] == "genmove" || words[0] == "analyze") && words.size() == 2) {
                    analyzed_game = session.game();
                    analysis = session.analyze(parse_integer(words[1]));
                    if (words[0] == "genmove") session.play(analysis->action);
                } else if (!(words[0] == "state" && words.size() == 1)) throw std::runtime_error("Unknown command or arguments");
                session.game();
                std::cout << "{\"ok\":true,\"state\":";
                write_state(session);
                if (analysis) {
                    std::cout << ",\"analysis\":";
                    write_analysis(*analysis, *analyzed_game, evaluator.requests.load() - requests, evaluator.batches.load() - batches);
                }
                std::cout << "}\n" << std::flush;
            } catch (const std::exception& error) { write_error(error.what()); }
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
