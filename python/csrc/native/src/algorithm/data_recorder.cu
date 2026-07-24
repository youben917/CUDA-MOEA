#include "cuda_moea/algorithm/data_recorder.cuh"

#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cuda_moea::detail {
namespace {

template <typename T>
std::vector<T> copy_from_device(const T* source, std::size_t count)
{
    if (count == 0) return {};
    if (!source) {
        throw std::invalid_argument(
            "Cannot save a non-empty array from a null device pointer");
    }

    std::vector<T> host(count);
    const cudaError_t status = cudaMemcpy(
        host.data(),
        source,
        count * sizeof(T),
        cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string("Data snapshot cudaMemcpy failed: ") +
            cudaGetErrorString(status));
    }
    return host;
}

template <typename T>
void write_binary(
    const std::filesystem::path& path,
    const T* values,
    std::size_t count)
{
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error(
            "Cannot create data file: " + path.string());
    }
    if (count > 0) {
        output.write(
            reinterpret_cast<const char*>(values),
            static_cast<std::streamsize>(count * sizeof(T)));
    }
    if (!output) {
        throw std::runtime_error(
            "Failed to write data file: " + path.string());
    }
}

template <typename T>
void write_binary(
    const std::filesystem::path& path,
    const std::vector<T>& values)
{
    write_binary(path, values.data(), values.size());
}

std::filesystem::path generation_directory(
    const std::filesystem::path& root,
    int generation)
{
    std::ostringstream name;
    name << "generation_" << std::setw(6) << std::setfill('0')
         << generation;
    return root / name.str();
}

std::string json_string(const std::string& value)
{
    std::ostringstream output;
    output << '"';
    for (const char c : value) {
        switch (c) {
        case '"': output << "\\\""; break;
        case '\\': output << "\\\\"; break;
        case '\b': output << "\\b"; break;
        case '\f': output << "\\f"; break;
        case '\n': output << "\\n"; break;
        case '\r': output << "\\r"; break;
        case '\t': output << "\\t"; break;
        default:
            if (static_cast<unsigned char>(c) < 0x20) {
                output << "\\u"
                       << std::hex << std::setw(4) << std::setfill('0')
                       << static_cast<int>(
                              static_cast<unsigned char>(c))
                       << std::dec << std::setfill(' ');
            } else {
                output << c;
            }
            break;
        }
    }
    output << '"';
    return output.str();
}

void ensure_directory_exists(const std::filesystem::path& path)
{
    std::error_code error;
    std::filesystem::create_directories(path, error);
    if (error) {
        throw std::runtime_error(
            "Cannot create data directory '" + path.string() +
            "': " + error.message());
    }
}

} // namespace

DataRecorder::DataRecorder(DataSaveConfig config)
    : config_(std::move(config)),
      root_(config_.output_directory)
{}

bool DataRecorder::enabled() const noexcept {
    return config_.enabled;
}

bool DataRecorder::should_save(
    int generation,
    int max_generations) const noexcept
{
    if (!enabled()) return false;
    return generation == 0 ||
        generation == max_generations ||
        generation % config_.generation_interval == 0;
}

void DataRecorder::initialize(
    const AlgorithmInfo& algorithm,
    const ProblemInfo& problem,
    const CudaConfig& cuda_config,
    ReferenceDirectionView references,
    CudaContext& cuda)
{
    if (!enabled()) return;
    ensure_directory_exists(root_);
    cuda.synchronize();

    write_binary(
        root_ / "lower_bounds.bin",
        problem.lower_bounds);
    write_binary(
        root_ / "upper_bounds.bin",
        problem.upper_bounds);

    if (references.initial_values &&
        references.objective_count > 0 &&
        references.count > 0) {
        const std::size_t count =
            static_cast<std::size_t>(references.objective_count) *
            references.count;
        write_binary(
            root_ / "initial_reference_directions.bin",
            copy_from_device(references.initial_values, count));
    }

    std::ofstream metadata(
        root_ / "metadata.json",
        std::ios::trunc);
    if (!metadata) {
        throw std::runtime_error(
            "Cannot create metadata.json in " + root_.string());
    }
    metadata
        << "{\n"
        << "  \"format_version\": 1,\n"
        << "  \"scalar_type\": \"float32\",\n"
        << "  \"index_type\": \"int32\",\n"
        << "  \"algorithm_name\": "
        << (algorithm.name.empty() ? "null" : json_string(algorithm.name))
        << ",\n"
        << "  \"problem_name\": "
        << (problem.name.empty() ? "null" : json_string(problem.name))
        << ",\n"
        << "  \"population_size\": " << algorithm.population_size << ",\n"
        << "  \"dimension\": " << algorithm.dimension << ",\n"
        << "  \"objective_count\": " << algorithm.objective_count << ",\n"
        << "  \"constraint_count\": " << problem.constraint_count << ",\n"
        << "  \"max_generations\": " << algorithm.max_generations << ",\n"
        << "  \"random_seed\": " << cuda_config.seed << ",\n"
        << "  \"generation_interval\": "
        << config_.generation_interval << ",\n"
        << "  \"reference_direction_count\": "
        << references.count << ",\n"
        << "  \"layouts\": {\n"
        << "    \"variables\": \"row-major (N,D)\",\n"
        << "    \"objectives\": \"objective-major (M,N)\",\n"
        << "    \"constraints\": \"(N)\",\n"
        << "    \"reference_directions\": \"objective-major (M,K)\",\n"
        << "    \"bounds\": \"separate lower/upper arrays (D)\"\n"
        << "  }\n"
        << "}\n";
    if (!metadata) {
        throw std::runtime_error(
            "Failed to write metadata.json in " + root_.string());
    }
    initialized_ = true;
}

void DataRecorder::save_snapshot(
    int generation,
    ConstPopulationView population,
    ReferenceDirectionView references,
    DeviceSpan<const int> auxiliary_indices,
    int active_count,
    CudaContext& cuda)
{
    if (!enabled()) return;
    if (!initialized_) {
        throw std::logic_error(
            "DataRecorder must be initialized before saving snapshots");
    }

    cuda.synchronize();
    const auto directory =
        generation_directory(root_, generation);
    ensure_directory_exists(directory);

    write_binary(
        directory / "variables.bin",
        copy_from_device(
            population.variables,
            static_cast<std::size_t>(population.size) *
                population.dimension));
    write_binary(
        directory / "objectives.bin",
        copy_from_device(
            population.objectives,
            static_cast<std::size_t>(population.objective_count) *
                population.size));
    write_binary(
        directory / "constraints.bin",
        copy_from_device(
            population.constraints,
            static_cast<std::size_t>(population.size)));

    if (references.values &&
        references.objective_count > 0 &&
        references.count > 0) {
        write_binary(
            directory / "reference_directions.bin",
            copy_from_device(
                references.values,
                static_cast<std::size_t>(references.objective_count) *
                    references.count));
    }

    if (!auxiliary_indices.empty()) {
        write_binary(
            directory / "selector_auxiliary_indices.bin",
            copy_from_device(
                auxiliary_indices.data,
                auxiliary_indices.size));
    }

    std::ofstream snapshot(
        directory / "snapshot.json",
        std::ios::trunc);
    if (!snapshot) {
        throw std::runtime_error(
            "Cannot create snapshot.json in " + directory.string());
    }
    snapshot
        << "{\n"
        << "  \"generation\": " << generation << ",\n"
        << "  \"population_size\": " << population.size << ",\n"
        << "  \"dimension\": " << population.dimension << ",\n"
        << "  \"objective_count\": " << population.objective_count << ",\n"
        << "  \"active_count\": " << active_count << ",\n"
        << "  \"reference_direction_count\": "
        << references.count << ",\n"
        << "  \"auxiliary_index_count\": "
        << auxiliary_indices.size << "\n"
        << "}\n";
    if (!snapshot) {
        throw std::runtime_error(
            "Failed to write snapshot.json in " + directory.string());
    }
}

} // namespace cuda_moea::detail
