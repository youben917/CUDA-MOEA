cmake_minimum_required(VERSION 3.24)
project(bi_sphere LANGUAGES CXX CUDA)
find_package(CudaMoeaNative CONFIG REQUIRED)
cuda_moea_add_problem(bi_sphere problem.cu registration.cpp)
