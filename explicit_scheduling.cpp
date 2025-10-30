#include <chrono>
#include <cstdio>
#include <omp.h>

using time_point = std::chrono::time_point<std::chrono::high_resolution_clock>;

struct Task {
    time_point start_time;
    size_t task_number;
    bool* initialization_done = nullptr;

    void initialization();
    void verification();
    void refinement();
    void all();
};

static const double MAX_TIME = 0.1;
static const size_t NTASKS = 11;
static std::atomic<bool> stop_requested;

static const bool DEFAULT_SCHEDULING = true;

int main(int, char**) {
    omp_set_num_threads(2);
    stop_requested = false;
    auto start = std::chrono::high_resolution_clock::now();

    // Create an array of tasks
    Task tasks[NTASKS];

    if (DEFAULT_SCHEDULING) {
#pragma omp parallel for shared(stop_requested)
        for (size_t i = 0; i < NTASKS; ++i) {
            if (stop_requested)
                continue;
            tasks[i] = Task{start, i};
            tasks[i].all();
        }
    } else {
#pragma omp parallel for shared(stop_requested)
        for (size_t i = 0; i < NTASKS; ++i) {
            if (stop_requested)
                continue;
            tasks[i] = Task{start, i};
            tasks[i].initialization();
        }

#pragma omp parallel for shared(stop_requested)
        for (size_t i = 0; i < NTASKS; ++i) {
            if (stop_requested)
                continue;
            tasks[i].verification();
        }

#pragma omp parallel for shared(stop_requested)
        for (size_t i = 0; i < NTASKS; ++i) {
            if (stop_requested)
                continue;
            tasks[i].refinement();
        }
    }

    auto inter_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = inter_time - start;
    std::printf("Total elapsed time: %.6f seconds\n", elapsed.count());

    // Check if all initializations completed
    for (size_t i = 0; i < NTASKS; ++i) {
        auto done = *tasks[i].initialization_done;
        std::printf("Task %zu initialization done: %s\n", i, done ? "true" : "false");

        delete tasks[i].initialization_done;
    }
}

double elapsed_time(time_point start_time) {
    auto inter_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = inter_time - start_time;
    return elapsed.count();
}

void Task::initialization() {
    auto current = elapsed_time(start_time);
    std::printf("Initialization %zu at %f\n", this->task_number, current);
    for (int i = 0; i < 10000; ++i) {
        auto current = elapsed_time(start_time);
        if (current > MAX_TIME) {
            stop_requested = true;
            std::printf("Time limit reached for task %zu during Initialization\n",
                        this->task_number);
            return;
        }
    }
    this->initialization_done = new bool(true);
    return;
}

void Task::verification() {
    auto current = elapsed_time(start_time);
    std::printf("Verification %zu at %f\n", this->task_number, current);
    for (int i = 0; i < 500000; ++i) {
        auto current = elapsed_time(start_time);
        if (current > MAX_TIME) {
            stop_requested = true;
            std::printf("Time limit reached for task %zu during Verification\n", this->task_number);
            return;
        }
    }
}

void Task::refinement() {
    auto current = elapsed_time(start_time);
    std::printf("Refinement %zu at %f\n", this->task_number, current);
    for (int i = 0; i < 100000; ++i) {
        auto current = elapsed_time(start_time);
        if (current > MAX_TIME) {
            stop_requested = true;
            std::printf("Time limit reached for task %zu during Refinement\n", this->task_number);
            return;
        }
    }
}

void Task::all() {
    if (stop_requested)
        return;
    this->initialization();
    if (stop_requested)
        return;
    this->verification();
    if (stop_requested)
        return;
    this->refinement();
}