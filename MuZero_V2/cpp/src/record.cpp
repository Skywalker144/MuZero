#include "muzero/record.h"
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <system_error>
#include <unistd.h>
#include <zlib.h>

namespace muzero {
namespace {
void integer(std::ostream& stream, uint32_t value) {
    char bytes[4];
    for (int i = 0; i < 4; ++i) bytes[i] = static_cast<char>(value >> (8 * i));
    stream.write(bytes, 4);
}
uint32_t integer(std::istream& stream) {
    unsigned char bytes[4];
    stream.read(reinterpret_cast<char*>(bytes), 4);
    uint32_t result = 0;
    for (int i = 0; i < 4; ++i) result |= uint32_t(bytes[i]) << (8 * i);
    return result;
}
void floating(std::ostream& stream, float value) {
    uint32_t bits;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    integer(stream, bits);
}
void sync_path(const std::filesystem::path& path) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::system_error(errno, std::generic_category(), "Open for sync");
    int result = ::fsync(fd), error = errno;
    ::close(fd);
    if (result != 0) throw std::system_error(error, std::generic_category(), "Sync record");
}
}

RecordWriter::RecordWriter(const std::filesystem::path& directory, size_t max_rows, size_t capacity, int canvas)
    : directory_(directory), max_rows_(max_rows), capacity_(capacity) {
    if (!max_rows || !capacity) throw std::runtime_error("Invalid writer limits");
    for (const auto& entry : std::filesystem::directory_iterator(directory_)) {
        if (entry.path().extension() != ".mzs") continue;
        std::ifstream input(entry.path(), std::ios::binary);
        input.exceptions(std::ios::badbit | std::ios::failbit);
        char magic[8];
        input.read(magic, 8);
        if (std::memcmp(magic, GAME_MAGIC, 8)) throw std::runtime_error("Shard protocol mismatch");
        auto count = integer(input);
        if (!count || count > std::filesystem::file_size(entry.path()) / 32)
            throw std::runtime_error("Invalid shard game count");
        for (uint32_t i = 0; i < count; ++i) {
            auto id = integer(input), stored_canvas = integer(input), size = integer(input), rule = integer(input);
            int winner = static_cast<int32_t>(integer(input));
            auto rows = integer(input), opening = integer(input), compressed = integer(input);
            if (stored_canvas != static_cast<unsigned>(canvas) || size < 5 || size > stored_canvas ||
                rule > static_cast<unsigned>(Rule::RENJU) || winner < -1 || winner > 1 ||
                opening > size * size || rows > size * size - opening || rows + opening == 0 ||
                !compressed || !committed_.insert(static_cast<int>(id)).second)
                throw std::runtime_error("Invalid or duplicate game in shard");
            Game prefix(size, canvas, static_cast<Rule>(rule));
            for (uint32_t i = 0; i < opening; ++i) prefix.play(static_cast<int>(integer(input)));
            if ((prefix.finished() && (rows != 0 || prefix.winner() != winner)) || (!prefix.finished() && rows == 0))
                throw std::runtime_error("Invalid opening termination");
            input.seekg(compressed, std::ios::cur);
        }
        if (static_cast<uint64_t>(input.tellg()) != std::filesystem::file_size(entry.path()))
            throw std::runtime_error("Invalid shard size");
        auto stem = entry.path().stem().string();
        if (stem.rfind("shard_", 0) != 0) throw std::runtime_error("Invalid shard filename");
        next_shard_ = std::max(next_shard_, static_cast<unsigned>(std::stoul(stem.substr(6))) + 1);
    }
    worker_ = std::thread(&RecordWriter::write_loop, this);
}

RecordWriter::~RecordWriter() {
    if (worker_.joinable()) {
        { std::lock_guard<std::mutex> lock(mutex_); closing_ = true; }
        changed_.notify_all();
        worker_.join();
    }
}

void RecordWriter::enqueue(FinishedGame game) {
    std::unique_lock<std::mutex> lock(mutex_);
    changed_.wait(lock, [&] { return queue_.size() < capacity_ || failure_ || closing_; });
    if (failure_) std::rethrow_exception(failure_);
    if (closing_) throw std::runtime_error("Writer is closed");
    queue_.push_back(std::move(game));
    changed_.notify_all();
}

void RecordWriter::finish() {
    { std::lock_guard<std::mutex> lock(mutex_); closing_ = true; }
    changed_.notify_all();
    if (worker_.joinable()) worker_.join();
    if (failure_) std::rethrow_exception(failure_);
}

void RecordWriter::write_loop() {
    try {
        std::vector<FinishedGame> batch;
        size_t rows = 0;
        auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
        while (true) {
            std::unique_lock<std::mutex> lock(mutex_);
            changed_.wait_until(lock, deadline, [&] { return !queue_.empty() || closing_; });
            if (!queue_.empty()) {
                rows += queue_.front().steps.size();
                batch.push_back(std::move(queue_.front()));
                queue_.pop_front();
                changed_.notify_all();
            }
            bool done = closing_ && queue_.empty();
            lock.unlock();
            if (rows >= max_rows_ || done || std::chrono::steady_clock::now() >= deadline) {
                if (!batch.empty()) publish(batch);
                batch.clear();
                rows = 0;
                deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
            }
            if (done) break;
        }
    } catch (...) {
        std::lock_guard<std::mutex> lock(mutex_);
        failure_ = std::current_exception();
        changed_.notify_all();
    }
}

void RecordWriter::publish(const std::vector<FinishedGame>& games) {
    std::ostringstream name;
    name << "shard_" << std::setw(8) << std::setfill('0') << next_shard_++ << ".mzs";
    auto destination = directory_ / name.str(), temporary = destination;
    temporary += ".tmp";
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    output.exceptions(std::ios::badbit | std::ios::failbit);
    output.write(GAME_MAGIC, 8);
    integer(output, games.size());
    size_t rows = 0;
    for (const auto& game : games) {
        std::ostringstream raw(std::ios::binary | std::ios::out);
        for (const auto& step : game.steps) {
            integer(raw, step.player);
            integer(raw, step.action);
            floating(raw, step.weight);
            integer(raw, step.budget.visits);
            for (size_t start = 0; start < step.observation.size(); start += 8) {
                unsigned char packed = 0;
                for (size_t bit = 0; bit < 8 && start + bit < step.observation.size(); ++bit)
                    if (step.observation[start + bit] != 0) packed |= 1 << (7 - bit);
                raw.put(static_cast<char>(packed));
            }
            for (double value : step.policy) floating(raw, value);
        }
        auto bytes = raw.str();
        uLongf length = compressBound(bytes.size());
        std::vector<unsigned char> compressed(length);
        if (compress2(compressed.data(), &length, reinterpret_cast<const Bytef*>(bytes.data()), bytes.size(), Z_BEST_SPEED) != Z_OK)
            throw std::runtime_error("Game compression failed");
        integer(output, game.id);
        integer(output, game.canvas);
        integer(output, game.size);
        integer(output, static_cast<uint32_t>(game.rule));
        integer(output, game.winner);
        integer(output, game.steps.size());
        integer(output, game.opening.size());
        integer(output, length);
        for (int action : game.opening) integer(output, action);
        output.write(reinterpret_cast<const char*>(compressed.data()), length);
        rows += game.steps.size();
    }
    output.flush();
    output.close();
    sync_path(temporary);
    std::filesystem::rename(temporary, destination);
    sync_path(directory_);
    std::cout << "shard=" << name.str() << " committed_games=" << games.size() << " rows=" << rows << std::endl;
}

}
