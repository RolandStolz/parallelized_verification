#include <chrono>
#include <cstdio>
#include <omp.h>

using time_point = std::chrono::time_point<std::chrono::high_resolution_clock>;
void task(int number, time_point start_time);

static const double MAX_TIME = 0.1;
static const size_t NTASKS = 11;

// This array is supposed to emphasize that not finishing the initialization can cause all kinds of
// problems (not just safety related)
static bool* INIT_RESULT[NTASKS] = {};
static std::atomic<bool> stop_requested;

int main(int, char**) {
    omp_set_num_threads(2);
    stop_requested = false;
    auto start = std::chrono::high_resolution_clock::now();

#pragma omp parallel for
    for (int i = 0; i < NTASKS; ++i) {
        if (stop_requested)
            continue;
        task(i, start);
    }

    auto inter_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = inter_time - start;
    std::printf("Total elapsed time: %.6f seconds\n", elapsed.count());

    // Check if all initializations completed
    for (size_t i = 0; i < NTASKS; ++i) {
        auto _ = *INIT_RESULT[i];
        delete INIT_RESULT[i];
    }
}

void task(int number, time_point start_time) {
    std::printf("Initialization %d\n", number);
    for (int i = 0; i < 10000; ++i) {
        auto inter_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = inter_time - start_time;
        if (elapsed.count() > MAX_TIME) {
            stop_requested = true;
            std::printf("Time limit reached for task %d during Initialization\n", number);
            return;
        }
    }
    INIT_RESULT[number] = new bool(true);

    std::printf("Verification %d\n", number);
    for (int i = 0; i < 500000; ++i) {
        auto inter_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = inter_time - start_time;
        if (elapsed.count() > MAX_TIME) {
            stop_requested = true;
            std::printf("Time limit reached for task %d during Verification\n", number);
            return;
        }
    }

    std::printf("Refinement %d\n", number);
    for (int i = 0; i < 100000; ++i) {
        auto inter_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = inter_time - start_time;
        if (elapsed.count() > MAX_TIME) {
            stop_requested = true;
            std::printf("Time limit reached for task %d during Refinement\n", number);
            return;
        }
    }
}