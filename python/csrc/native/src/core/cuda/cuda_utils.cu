#include <iostream>
#include <iomanip>  // for std::setw and std::left

size_t get_device_mem_threshold(int device) {
    
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    
    size_t threshold = static_cast<size_t>(free_mem * 0.995);

    constexpr double gb_unit = 1024.0 * 1024.0 * 1024.0;
    double total_gb     = static_cast<double>(total_mem) / gb_unit;
    double free_gb      = static_cast<double>(free_mem)  / gb_unit;
    double threshold_gb = static_cast<double>(threshold) / gb_unit;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Device " << device
              << " - Total: " << total_gb << " GB, "
              << "Free: " << free_gb << " GB, "
              << "Pool threshold: " << threshold_gb << " GB" << std::endl;

    return threshold;
}
