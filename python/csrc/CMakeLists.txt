find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(Torch REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python
    PATHS "${TORCH_INSTALL_PREFIX}/lib" REQUIRED NO_DEFAULT_PATH)

Python_add_library(_C MODULE WITH_SOABI bindings.cpp adapters.cpp native_problem.cpp)
target_include_directories(_C PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CUDA_MOEA_NATIVE_ROOT}/include"
)
target_link_libraries(_C PRIVATE cuda_moea_native ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY} ${CMAKE_DL_LIBS})
target_compile_features(_C PRIVATE cxx_std_17)
set_property(TARGET _C PROPERTY POSITION_INDEPENDENT_CODE ON)

if(TORCH_CXX_FLAGS)
    separate_arguments(CUDA_MOEA_TORCH_FLAGS NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
    target_compile_options(_C PRIVATE ${CUDA_MOEA_TORCH_FLAGS})
    target_compile_options(cuda_moea_native PRIVATE ${CUDA_MOEA_TORCH_FLAGS})
endif()

if(NOT CUDA_MOEA_PYTHON_OUTPUT_DIR)
    set(CUDA_MOEA_PYTHON_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/package")
endif()
set_target_properties(_C PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CUDA_MOEA_PYTHON_OUTPUT_DIR}")

set_target_properties(cuda_moea_native PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CUDA_MOEA_PYTHON_OUTPUT_DIR}/lib"
    BUILD_RPATH_USE_ORIGIN ON)
set_property(TARGET _C APPEND PROPERTY BUILD_RPATH "$ORIGIN/lib")

# An exact SDK fingerprint deliberately invalidates independently built plugins
# when the host-side C++ contract, core implementation or toolchain changes.
file(GLOB_RECURSE sdk_inputs CONFIGURE_DEPENDS
    "${CUDA_MOEA_NATIVE_ROOT}/include/*" "${CUDA_MOEA_NATIVE_ROOT}/src/*")
list(APPEND sdk_inputs "${CMAKE_CURRENT_SOURCE_DIR}/native_problem.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/native_sdk.cmake.in"
    "${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt" "${PROJECT_SOURCE_DIR}/CMakeLists.txt")
list(SORT sdk_inputs)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${sdk_inputs})
set(sdk_fingerprint "${PROJECT_VERSION};${CMAKE_CXX_COMPILER_ID};${CMAKE_CXX_COMPILER_VERSION};${CMAKE_CUDA_COMPILER_VERSION};${TORCH_CXX_FLAGS};${CMAKE_CXX_FLAGS};${CMAKE_CUDA_FLAGS};${CMAKE_CXX_FLAGS_DEBUG};${CMAKE_CXX_FLAGS_RELEASE};${CMAKE_CUDA_FLAGS_DEBUG};${CMAKE_CUDA_FLAGS_RELEASE};${CMAKE_BUILD_TYPE};${CMAKE_SYSTEM_PROCESSOR}")
foreach(input IN LISTS sdk_inputs)
    file(SHA256 "${input}" digest)
    string(APPEND sdk_fingerprint ";${digest}")
endforeach()
string(SHA256 CUDA_MOEA_BUILD_ID "${sdk_fingerprint}")
set(sdk "${CUDA_MOEA_PYTHON_OUTPUT_DIR}/sdk")
get_target_property(sdk_architectures cuda_moea_native CUDA_ARCHITECTURES)
file(MAKE_DIRECTORY "${sdk}/include/cuda_moea/native" "${sdk}/cmake")
file(COPY "${CUDA_MOEA_NATIVE_ROOT}/include/" DESTINATION "${sdk}/include")
file(CONFIGURE OUTPUT "${sdk}/include/cuda_moea/native/build_id.h"
    CONTENT "#pragma once\n#define CUDA_MOEA_NATIVE_BUILD_ID \"@CUDA_MOEA_BUILD_ID@\"\n" @ONLY)
file(CONFIGURE OUTPUT "${sdk}/build.json"
    CONTENT "{\"build_id\":\"@CUDA_MOEA_BUILD_ID@\",\"architectures\":\"@sdk_architectures@\",\"abi\":1}\n" @ONLY)
configure_file(native_sdk.cmake.in "${sdk}/cmake/CudaMoeaNativeConfig.cmake" @ONLY)
target_include_directories(_C PRIVATE "${sdk}/include")
