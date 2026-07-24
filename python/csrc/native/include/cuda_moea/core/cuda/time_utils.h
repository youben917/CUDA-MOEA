#pragma once

#include <chrono>
#include <iostream>
#include <iomanip>
#include <string>
#include <cctype>

#include "cuda_utils.cuh"
/**
 * @brief Timer class for simplified timing operations
 * 
 * Provides high-resolution timing functionality with automatic initialization
 * and easy-to-use interface for measuring elapsed time in milliseconds.
 */
class Timer {
private:
    std::chrono::high_resolution_clock::time_point start_time_;
    
public:
    /**
     * @brief Constructor that automatically starts timing
     */
    Timer() {
        start();
    }
    
    /**
     * @brief Start or restart the timer
     */
    void start() {
        start_time_ = std::chrono::high_resolution_clock::now();
    }
    
    /**
     * @brief Get elapsed time in milliseconds
     * @return Elapsed time since start() was called, in milliseconds
     */
    double elapsed_ms() const {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time_);
        return duration.count() / 1000.0;
    }
};

/**
 * @brief Template function to time the execution of any callable
 * 
 * @tparam Func Callable type (function, lambda, etc.)
 * @param func_name Name of the function for display purposes
 * @param func The callable to be timed
 * @param print_immediately Whether to print timing result immediately
 * @return Elapsed time in milliseconds
 */
template<typename Func>
inline double time_function(const std::string& func_name, Func&& func, bool print_immediately = true) {
    Timer timer;
    func();
    // CUDA_CHECK(cudaDeviceSynchronize()); // Ensure all GPU operations are completed
    double elapsed = timer.elapsed_ms();
    
    if (print_immediately) {
        std::cout << std::left << std::setw(40) << func_name 
                  << std::setw(15) << std::fixed << std::setprecision(3) << elapsed << std::endl;
    }
    
    return elapsed;
}

/**
 * @brief Initialize timing output format with headers
 * 
 * Prints a formatted header for timing statistics output.
 */

inline void nsga_iteration_start_output(const std::string& algo_type) {
    std::string algo_upper = algo_type;
    for (char& ch : algo_upper) {
        ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
    }

    if (algo_upper != "NSGA3" && algo_upper != "RVEA") {
        algo_upper = "NSGA3";
    }

    std::cout << "\n=== CUDA-" << algo_upper << " Iterater Starts ===" << std::endl;
}

inline void init_timing_output() {
    std::cout << "\n=== Detailed Function Execution Time Statistics ===" << std::endl;
    std::cout << std::left << std::setw(40) << "Function Name" << std::setw(15) << "Time (ms)" << std::endl;
    std::cout << std::string(50, '-') << std::endl;
}

/**
 * @brief Print total execution time with formatting
 * 
 * @param total_time_ms Total execution time in milliseconds
 */
inline void print_total_time(double total_time_ms) {
    std::cout << std::string(50, '-') << std::endl;
    std::cout << std::left << std::setw(40) << "Total NSGA-3 Iteration Execution Time" 
              << std::setw(15) << std::fixed << std::setprecision(3) << total_time_ms << std::endl;
    std::cout << std::string(50, '=') << std::endl;
}
