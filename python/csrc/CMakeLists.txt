find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(Torch REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python
    PATHS "${TORCH_INSTALL_PREFIX}/lib" REQUIRED NO_DEFAULT_PATH)

Python_add_library(_C MODULE WITH_SOABI bindings.cpp adapters.cpp)
target_include_directories(_C PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CUDA_MOEA_NATIVE_ROOT}/include"
)
target_link_libraries(_C PRIVATE cuda_moea_native ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})
target_compile_features(_C PRIVATE cxx_std_17)
set_property(TARGET _C PROPERTY POSITION_INDEPENDENT_CODE ON)

if(TORCH_CXX_FLAGS)
    separate_arguments(CUDA_MOEA_TORCH_FLAGS NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
    target_compile_options(_C PRIVATE ${CUDA_MOEA_TORCH_FLAGS})
    target_compile_options(cuda_moea_native PRIVATE ${CUDA_MOEA_TORCH_FLAGS})
endif()

if(CUDA_MOEA_PYTHON_OUTPUT_DIR)
    set_target_properties(_C PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CUDA_MOEA_PYTHON_OUTPUT_DIR}")
endif()
