#include "muzero/match.h"
#include "muzero/torch_backend.h"
#include <ATen/Parallel.h>
#include <csignal>
#include <iomanip>
#include <iostream>
#include <set>

using namespace muzero;

namespace {
static_assert(std::atomic<bool>::is_always_lock_free);
std::atomic<bool> stopped{false};
void stop(int) { stopped.store(true, std::memory_order_relaxed); }

struct Task {
    int id, generator, mask;
    uint64_t seed;
    bool generated;
    std::vector<int> moves;
};

void write_moves(std::ostream& out, const std::vector<int>& moves, int canvas, int size) {
    out << '[';
    for (size_t i = 0; i < moves.size(); ++i) {
        if (i) out << ',';
        out << moves[i] / canvas * size + moves[i] % canvas;
    }
    out << ']';
}

class Workers {
    int count_;
public:
    explicit Workers(int count) : count_(count) {
        if (count < 1) throw std::runtime_error("Invalid worker count");
    }
    void run(int total, const std::function<void(int)>& work) const {
        std::atomic<int> next{0};
        std::mutex mutex;
        std::exception_ptr failure;
        std::vector<std::thread> workers;
        try {
            for (int i = 0; i < std::min(total, count_); ++i) workers.emplace_back([&] {
                try {
                    while (!stopped) {
                        int index = next++;
                        if (index >= total) break;
                        work(index);
                    }
                } catch (...) {
                    stopped = true;
                    std::lock_guard<std::mutex> lock(mutex);
                    if (!failure) failure = std::current_exception();
                }
            });
        } catch (...) {
            stopped = true;
            for (auto& worker : workers) worker.join();
            throw;
        }
        for (auto& worker : workers) worker.join();
        if (failure) std::rethrow_exception(failure);
    }
};
}

int main(int argc, char** argv) {
    try {
        if (argc != 5) throw std::runtime_error("Usage: muzero_match resolved.cfg model_a.pt model_b.pt tasks.txt");
        Config raw(argv[1]);
        SearchConfig search(raw, SearchProfile::Evaluation);
        OpeningConfig opening(raw);
        int size = raw.integer("BOARD_SIZE");
        Rule rule = parse_rule(raw.text("RULE"));
        if (opening.probability != 1 || opening.rejection_probability_fallback >= 1)
            throw std::runtime_error("Matches require balanced openings with a rejection fallback below 1");
        at::set_num_threads(raw.integer("TORCH_THREADS"));
        auto backend_a = std::make_unique<TorchBackend>(argv[2], raw.text("DEVICE"), 0, raw.number("NN_POLICY_TEMPERATURE"));
        int canvas = backend_a->canvas_size();
        auto backend_b = std::make_unique<TorchBackend>(argv[3], raw.text("DEVICE"), canvas, raw.number("NN_POLICY_TEMPERATURE"));
        Game(size, canvas, rule);
        BatchEvaluator a(std::move(backend_a), raw.integer("NN_MAX_BATCH_SIZE"), raw.integer("NN_BATCH_WAIT_US"));
        BatchEvaluator b(std::move(backend_b), raw.integer("NN_MAX_BATCH_SIZE"), raw.integer("NN_BATCH_WAIT_US"));
        std::ifstream input(argv[4]);
        if (!input) throw std::runtime_error("Cannot read match tasks");
        std::vector<Task> tasks;
        std::set<int> ids;
        for (std::string line; std::getline(input, line);) {
            std::istringstream row(line);
            Task task{};
            int count;
            if (!(row >> task.id >> task.seed >> task.generator >> task.mask >> count) || task.id < 0 ||
                task.generator != task.id % 2 || task.mask < 1 || task.mask > 3 || count < -1 || count >= size * size ||
                !ids.insert(task.id).second)
                throw std::runtime_error("Invalid match task");
            task.generated = count >= 0;
            Game replay(size, canvas, rule);
            for (int i = 0; i < count; ++i) {
                int move;
                if (!(row >> move) || move < 0 || move >= size * size) throw std::runtime_error("Invalid opening move");
                int action = move / size * canvas + move % size;
                replay.play(action);
                task.moves.push_back(action);
            }
            if (!(row >> std::ws).eof() || replay.finished()) throw std::runtime_error("Invalid opening task");
            tasks.push_back(std::move(task));
        }
        std::signal(SIGINT, stop);
        std::signal(SIGTERM, stop);
        Workers workers(raw.integer("NUM_GAME_THREADS"));
        std::mutex output;
        workers.run(static_cast<int>(tasks.size()), [&](int index) {
            auto& task = tasks[index];
            if (task.generated) return;
            auto result = match_opening(size, canvas, rule, opening, task.generator == 0 ? a : b, task.seed, stopped);
            if (!result) return;
            task.moves = result->actions;
            task.generated = true;
            std::ostringstream row;
            row << std::setprecision(17) << "{\"type\":\"opening\",\"id\":" << task.id
                << ",\"generator\":" << task.generator << ",\"seed\":" << task.seed
                << ",\"attempts\":" << result->attempts << ",\"value\":" << result->start_value << ",\"moves\":";
            write_moves(row, task.moves, canvas, size);
            row << '}';
            std::lock_guard<std::mutex> lock(output);
            std::cout << row.str() << '\n' << std::flush;
        });
        std::vector<std::pair<size_t, int>> games;
        for (size_t i = 0; i < tasks.size(); ++i) if (tasks[i].generated)
            for (int color = 0; color < 2; ++color) if (tasks[i].mask & (1 << color)) games.emplace_back(i, color);
        workers.run(static_cast<int>(games.size()), [&](int index) {
            auto [task_index, color] = games[index];
            const auto& task = tasks[task_index];
            Game game(size, canvas, rule);
            for (int action : task.moves) game.play(action);
            auto result = play_match(game, search, a, b, color == 0,
                                     {raw.integer("NUM_SEARCH_THREADS"), raw.number("VIRTUAL_LOSS"), &stopped},
                                     task.seed ^ (0xd1b54a32d192ed03ULL * (color + 1)));
            if (!result) return;
            std::vector<int> moves = task.moves;
            moves.insert(moves.end(), result->moves.begin(), result->moves.end());
            std::ostringstream row;
            row << std::setprecision(17) << "{\"type\":\"game\",\"id\":" << 2 * task.id + color
                << ",\"opening_id\":" << task.id << ",\"black_a\":" << (color == 0 ? "true" : "false")
                << ",\"winner\":" << result->winner << ",\"seconds\":" << result->seconds << ",\"moves\":";
            write_moves(row, moves, canvas, size);
            row << '}';
            std::lock_guard<std::mutex> lock(output);
            std::cout << row.str() << '\n' << std::flush;
        });
        std::cerr << "requests_a=" << a.requests << " batches_a=" << a.batches
                  << " requests_b=" << b.requests << " batches_b=" << b.batches << '\n';
        return stopped ? 130 : 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
