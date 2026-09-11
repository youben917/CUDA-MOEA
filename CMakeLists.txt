cmake_minimum_required(VERSION 3.24)
project(cuda_moea VERSION 0.1.0 LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)

if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "89" CACHE STRING
        "CUDA architectures for CUDA-MOEA")
endif()

set(CUDA_MOEA_NATIVE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/python/csrc/native")
set(CUDA_MOEA_PRIVATE_INCLUDE_DIRS
    "${CUDA_MOEA_NATIVE_ROOT}/src"
    "${CUDA_MOEA_NATIVE_ROOT}/src/core"
    "${CUDA_MOEA_NATIVE_ROOT}/src/core/cuda"
    "${CUDA_MOEA_NATIVE_ROOT}/src/problem/dtlz"
    "${CUDA_MOEA_NATIVE_ROOT}/src/reference"
    "${CUDA_MOEA_NATIVE_ROOT}/src/selection/nsga3/kernels"
    "${CUDA_MOEA_NATIVE_ROOT}/src/selection/rvea/kernels"
)

set(CUDA_MOEA_CORE_SOURCES
    ${CUDA_MOEA_NATIVE_ROOT}/src/core/cuda_context.cu
    ${CUDA_MOEA_NATIVE_ROOT}/src/core/population.cu
    ${CUDA_MOEA_NATIVE_ROOT}/src/core/cuda/cuda_utils.cu
    ${CUDA_MOEA_NATIVE_ROOT}/src/core/cuda/cuda_manager.cu
    ${CUDA_MOEA_NATIVE_ROOT}/src/core/cuda/cuda_random.cu
    ${CUDA_MOEA_NATIVE_ROOT}/src/core/cuda/cuda_warmup.cu
    ${CUDA_MOEA_NATIVE_ROOT}/src/core/cuda/cuda_vecops.cu
    ${CUDA_MOEA_NATIVE_ROOT}/src/core/cuda/cuda_atomops.cu
    ${CUDA_MOEA_NATIVE_ROOT}/src/core/cuda/cuda_ref_segment_ops.cu
)

set(CUDA_MOEA_PROBLEM_SOURCES
    ${CUDA_MOEA_NATIVE_ROOT}/src/problem/dtlz/dtlz_problem.cu
    ${CUDA_MOEA_NATIVE_ROOT}/src/problem/dtlz/dtlz_kernels.cu
    ${CUDA_MOEA_NATIVE_ROOT}/src/problem/dtlz/evaluation_workspace.cu
    ${CUDA_MOEA_NATIVE_ROOT}/src/problem/dtlz/population_adapter.cu
)

set(CUDA_MOEA_REPRODUCTION_SOURCES
    ${CUDA_MOEA_NATIVE_ROOT}/src/reproduction/mating_selection.cu
    ${CUDA_MOEA_NATIVE_ROOT}/src/reproduction/sbx_crossover.cu
    ${CUDA_MOEA_NATIVE_ROOT}/src/reproduction/polynomial_mutation.cu
    ${CUDA_MOEA_NATIVE_ROOT}/src/reproduction/reproduction_pipeline.cu
)

set(CUDA_MOEA_REFERENCE_SOURCES
    ${CUDA_MOEA_NATIVE_ROOT}/src/reference/reference_directions.cu
    ${CUDA_MOEA_NATIVE_ROOT}/src/reference/nsga3_reference_generator.cu
    ${CUDA_MOEA_NATIVE_ROOT}/src/reference/rvea_reference_generator.cu
    ${CUDA_MOEA_NATIVE_ROOT}/src/reference/rvea_reference_adaptation.cu
)

set(CUDA_MOEA_NSGA3_SELECTION_SOURCES
    ${CUDA_MOEA_NATIVE_ROOT}/src/selection/nsga3/nsga3_environment_selector.cu
    ${CUDA_MOEA_NATIVE_ROOT}/src/selection/nsga3/kernels/association.cu
    ${CUDA_MOEA_NATIVE_ROOT}/src/selection/nsga3/kernels/cv_quantization.cu
    ${CUDA_MOEA_NATIVE_ROOT}/src/selection/nsga3/kernels/dynamic_workspace.cu
    ${CUDA_MOEA_NATIVE_ROOT}/src/selection/nsga3/kernels/ideal_point.cu
    ${CUDA_MOEA_NATIVE_ROOT}/src/selection/nsga3/kernels/niching.cu
    ${CUDA_MOEA_NATIVE_ROOT}/src/selection/nsga3/kernels/non_dominated_sort.cu
    ${CUDA_MOEA_NATIVE_ROOT}/src/selection/nsga3/kernels/normalization.cu
    ${CUDA_MOEA_NATIVE_ROOT}/src/selection/nsga3/kernels/nsga3_population_update.cu
)

set(CUDA_MOEA_RVEA_SELECTION_SOURCES
    ${CUDA_MOEA_NATIVE_ROOT}/src/selection/rvea/rvea_environment_selector.cu
    ${CUDA_MOEA_NATIVE_ROOT}/src/selection/rvea/kernels/angle_penalized_distance.cu
    ${CUDA_MOEA_NATIVE_ROOT}/src/selection/rvea/kernels/elitism_selection.cu
    ${CUDA_MOEA_NATIVE_ROOT}/src/selection/rvea/kernels/front_zero.cu
    ${CUDA_MOEA_NATIVE_ROOT}/src/selection/rvea/kernels/objective_translation.cu
    ${CUDA_MOEA_NATIVE_ROOT}/src/selection/rvea/kernels/rvea_population_update.cu
    ${CUDA_MOEA_NATIVE_ROOT}/src/selection/rvea/kernels/reference_partition.cu
)

add_library(cuda_moea_native SHARED
    ${CUDA_MOEA_NATIVE_ROOT}/src/algorithm/algorithm.cu
    ${CUDA_MOEA_NATIVE_ROOT}/src/algorithm/data_recorder.cu
    ${CUDA_MOEA_NATIVE_ROOT}/src/builder/algorithm_builder.cu
    ${CUDA_MOEA_NATIVE_ROOT}/src/builder/nsga3_builder.cu
    ${CUDA_MOEA_NATIVE_ROOT}/src/builder/rvea_builder.cu
    ${CUDA_MOEA_NATIVE_ROOT}/src/factory/algorithm_factory.cu
    ${CUDA_MOEA_NATIVE_ROOT}/src/factory/operator_factory.cu
    ${CUDA_MOEA_NATIVE_ROOT}/src/factory/problem_factory.cu
    ${CUDA_MOEA_CORE_SOURCES}
    ${CUDA_MOEA_PROBLEM_SOURCES}
    ${CUDA_MOEA_REPRODUCTION_SOURCES}
    ${CUDA_MOEA_REFERENCE_SOURCES}
    ${CUDA_MOEA_NSGA3_SELECTION_SOURCES}
    ${CUDA_MOEA_RVEA_SELECTION_SOURCES}
)
target_include_directories(cuda_moea_native
    PRIVATE
        "${CUDA_MOEA_NATIVE_ROOT}/include"
        ${CUDA_MOEA_PRIVATE_INCLUDE_DIRS}
)

target_link_libraries(cuda_moea_native
    PRIVATE
        CUDA::cudart
        CUDA::cublas
        CUDA::cusolver
        OpenMP::OpenMP_CXX
)

target_compile_options(cuda_moea_native PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-extended-lambda
        --expt-relaxed-constexpr
        -O3
        -lineinfo
        -Xcompiler=-fopenmp
    >
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

set_target_properties(cuda_moea_native PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
)

add_subdirectory(python/csrc)
