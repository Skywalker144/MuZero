#pragma once

#include "search.h"
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <mutex>
#include <set>
#include <thread>

namespace muzero {

struct Step {
    int player, action;
    Budget budget;
    std::vector<float> observation;
    std::vector<double> policy;
};

struct FinishedGame {
    int id, canvas, size;
    Rule rule;
    int winner;
    std::vector<Step> steps;
    std::vector<int> opening;
};

class RecordWriter {
    std::filesystem::path directory_;
    size_t max_rows_, capacity_;
    unsigned next_shard_ = 0;
    std::set<int> committed_;
    std::mutex mutex_;
    std::condition_variable changed_;
    std::deque<FinishedGame> queue_;
    bool closing_ = false;
    std::exception_ptr failure_;
    std::thread worker_;
    void write_loop();
    void publish(const std::vector<FinishedGame>& games);
public:
    RecordWriter(const std::filesystem::path& directory, size_t max_rows, size_t capacity, int canvas);
    ~RecordWriter();
    const std::set<int>& committed() const { return committed_; }
    void enqueue(FinishedGame game);
    void finish();
};

}
