#include "adapters.h"
#include "native_problem.h"

#include <algorithm>
#include <chrono>
#include <climits>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include "cuda_moea/algorithms/nsga3.cuh"
#include "cuda_moea/algorithms/rvea.cuh"
#include "cuda_moea/factory/operator_factory.cuh"
#include "cuda_moea/factory/problem_factory.cuh"

namespace cuda_moea::python {
namespace {

std::string type_of(const py::dict& spec) {
    return py::cast<std::string>(spec["type"]);
}

int current_device() {
    int device = 0;
    const auto status = cudaGetDevice(&device);
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    return device;
}

torch::TensorOptions float_options(int device) {
    return torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, device);
}

torch::TensorOptions int_options(int device) {
    return torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, device);
}

torch::Tensor view_float(float* data, std::vector<int64_t> shape, int device) {
    return torch::from_blob(data, std::move(shape), [](void*) {}, float_options(device));
}

torch::Tensor view_float(const float* data, std::vector<int64_t> shape, int device) {
    return view_float(const_cast<float*>(data), std::move(shape), device);
}

torch::Tensor view_int(const int* data, std::vector<int64_t> shape, int device) {
    return torch::from_blob(const_cast<int*>(data), std::move(shape), [](void*) {}, int_options(device));
}

py::dict generation_context(const GenerationContext& context, int device) {
    py::dict result;
    result["generation"] = context.generation;
    result["max_generations"] = context.max_generations;
    result["device"] = device;
    return result;
}

py::dict evaluation_context(const EvaluationContext& context, int device) {
    py::dict result;
    result["generation"] = context.generation;
    result["max_generations"] = context.max_generations;
    result["device"] = device;
    return result;
}

py::dict algorithm_info(const AlgorithmInfo& info, int device) {
    py::dict result;
    result["name"] = info.name;
    result["population_size"] = info.population_size;
    result["dimension"] = info.dimension;
    result["objective_count"] = info.objective_count;
    result["max_generations"] = info.max_generations;
    result["device"] = device;
    return result;
}

py::dict population_dict(ConstPopulationView p, int device) {
    py::dict result;
    result["variables"] = view_float(p.variables, {p.size, p.dimension}, device);
    result["objectives"] = view_float(
        p.objectives, {p.objective_count, p.size}, device).transpose(0, 1);
    result["constraints"] = view_float(p.constraints, {p.size}, device);
    result["bounds"] = view_float(p.bounds, {p.dimension, 2}, device);
    return result;
}

py::dict population_dict(PopulationView p, int device) {
    return population_dict(static_cast<ConstPopulationView>(p), device);
}

c10::cuda::CUDAStreamGuard stream_guard(cudaStream_t stream, int device) {
    return c10::cuda::CUDAStreamGuard(
        c10::cuda::getStreamFromExternal(stream, static_cast<c10::DeviceIndex>(device)));
}

torch::Tensor checked_tensor(py::handle value, const char* name, int device,
                             torch::ScalarType dtype) {
    torch::Tensor tensor;
    try {
        tensor = py::cast<torch::Tensor>(value);
    } catch (const py::cast_error&) {
        throw std::invalid_argument(std::string(name) + " must be a torch.Tensor");
    }
    if (!tensor.is_cuda() || tensor.get_device() != device) {
        throw std::invalid_argument(std::string(name) + " must be on the algorithm CUDA device");
    }
    return tensor.to(dtype).contiguous();
}

void copy_evaluation(py::handle output, PopulationView population, int device) {
    py::handle objectives = output;
    py::handle constraints;
    if (py::isinstance<py::dict>(output)) {
        auto dict = py::reinterpret_borrow<py::dict>(output);
        objectives = dict["objectives"];
        if (dict.contains("constraints")) constraints = dict["constraints"];
    } else if (py::isinstance<py::tuple>(output)) {
        auto tuple = py::reinterpret_borrow<py::tuple>(output);
        objectives = tuple[0];
        if (tuple.size() > 1) constraints = tuple[1];
    }
    auto source = checked_tensor(objectives, "objectives", device, torch::kFloat32);
    if (source.dim() != 2 || source.size(0) != population.size ||
        source.size(1) != population.objective_count) {
        throw std::invalid_argument("objectives must have shape (population_size, objective_count)");
    }
    auto target = view_float(population.objectives,
        {population.objective_count, population.size}, device);
    target.copy_(source.transpose(0, 1));
    auto cv = view_float(population.constraints, {population.size}, device);
    if (constraints && !constraints.is_none()) {
        auto source_cv = checked_tensor(constraints, "constraints", device, torch::kFloat32);
        if (source_cv.numel() != population.size) {
            throw std::invalid_argument("constraints must contain population_size values");
        }
        cv.copy_(source_cv.reshape({population.size}));
    } else {
        cv.zero_();
    }
}

class PythonProblem final : public IProblemEvaluator {
public:
    explicit PythonProblem(const py::dict& spec) : object_(spec["object"]) {
        info_.name = py::cast<std::string>(spec["name"]);
        info_.dimension = py::cast<int>(spec["dimension"]);
        info_.objective_count = py::cast<int>(spec["objectives"]);
        info_.constraint_count = py::cast<int>(spec["constraints"]);
        info_.lower_bounds = py::cast<std::vector<float>>(spec["lower_bounds"]);
        info_.upper_bounds = py::cast<std::vector<float>>(spec["upper_bounds"]);
    }
    const ProblemInfo& info() const noexcept override { return info_; }
    void initialize(CudaContext& cuda) override {
        device_ = current_device();
        py::gil_scoped_acquire gil;
        if (py::hasattr(object_, "initialize")) {
            py::dict context; context["device"] = device_;
            object_.attr("initialize")(context);
        }
    }
    void evaluate(PopulationView p, const EvaluationContext& context,
                  cudaStream_t stream) override {
        auto guard = stream_guard(stream, device_);
        py::gil_scoped_acquire gil;
        auto output = object_.attr("evaluate")(
            view_float(p.variables, {p.size, p.dimension}, device_),
            evaluation_context(context, device_));
        copy_evaluation(output, p, device_);
    }
    bool prepare_parent(PopulationView p, const EvaluationContext& context,
                        cudaStream_t stream) override {
        py::gil_scoped_acquire gil;
        if (!py::hasattr(object_, "prepare_parent")) return false;
        auto guard = stream_guard(stream, device_);
        auto output = object_.attr("prepare_parent")(
            population_dict(p, device_), evaluation_context(context, device_));
        if (output.is_none()) return false;
        if (py::isinstance<py::bool_>(output)) return py::cast<bool>(output);
        copy_evaluation(output, p, device_);
        return true;
    }
    void reset() override {
        py::gil_scoped_acquire gil;
        if (py::hasattr(object_, "reset")) object_.attr("reset")();
    }
private:
    py::object object_;
    ProblemInfo info_;
    int device_ = 0;
};

class PythonMating final : public IMatingSelector {
public:
    explicit PythonMating(py::object object) : object_(std::move(object)) {}
    void initialize(const AlgorithmInfo& info, CudaContext& cuda) override {
        device_ = current_device();
        auto guard = stream_guard(cuda.execution_stream(), device_);
        py::gil_scoped_acquire gil;
        if (py::hasattr(object_, "initialize"))
            object_.attr("initialize")(algorithm_info(info, device_));
    }
    void select(ConstPopulationView parents, MatingStateView state,
                DeviceSpan<int> indices, const GenerationContext& context,
                cudaStream_t stream) override {
        auto guard = stream_guard(stream, device_);
        py::gil_scoped_acquire gil;
        py::dict state_dict;
        if (!state.rank.empty()) state_dict["rank"] = view_int(state.rank.data, {(long)state.rank.size}, device_);
        if (!state.reference_index.empty()) state_dict["reference_index"] = view_int(state.reference_index.data, {(long)state.reference_index.size}, device_);
        if (!state.score.empty()) state_dict["score"] = view_float(state.score.data, {(long)state.score.size}, device_);
        state_dict["active_count"] = state.active_count;
        auto output = object_.attr("select")(
            population_dict(parents, device_), state_dict,
            generation_context(context, device_));
        auto source = checked_tensor(output, "parent_indices", device_, torch::kInt32);
        if (source.numel() != static_cast<long>(indices.size)) {
            throw std::invalid_argument("parent_indices has the wrong number of elements");
        }
        view_int(indices.data, {(long)indices.size}, device_).copy_(source.reshape({(long)indices.size}));
    }
private:
    py::object object_;
    int device_ = 0;
};

class PythonCrossover final : public ICrossoverOperator {
public:
    explicit PythonCrossover(py::object object) : object_(std::move(object)) {}
    void initialize(const AlgorithmInfo& info, CudaContext& cuda) override {
        device_ = current_device();
        auto guard = stream_guard(cuda.execution_stream(), device_);
        py::gil_scoped_acquire gil;
        if (py::hasattr(object_, "initialize"))
            object_.attr("initialize")(algorithm_info(info, device_));
    }
    void apply(ConstPopulationView parents, DeviceSpan<const int> indices,
               PopulationView offspring, const GenerationContext& context,
               cudaStream_t stream) override {
        auto guard = stream_guard(stream, device_);
        py::gil_scoped_acquire gil;
        auto output = object_.attr("apply")(
            population_dict(parents, device_),
            view_int(indices.data, {(long)indices.size}, device_),
            generation_context(context, device_));
        auto source = checked_tensor(output, "offspring variables", device_, torch::kFloat32);
        if (source.sizes() != torch::IntArrayRef({offspring.size, offspring.dimension})) {
            throw std::invalid_argument("crossover output must have shape (population_size, dimension)");
        }
        view_float(offspring.variables, {offspring.size, offspring.dimension}, device_).copy_(source);
    }
private:
    py::object object_;
    int device_ = 0;
};

class PythonMutation final : public IMutationOperator {
public:
    explicit PythonMutation(py::object object) : object_(std::move(object)) {}
    void initialize(const AlgorithmInfo& info, CudaContext& cuda) override {
        device_ = current_device();
        auto guard = stream_guard(cuda.execution_stream(), device_);
        py::gil_scoped_acquire gil;
        if (py::hasattr(object_, "initialize"))
            object_.attr("initialize")(algorithm_info(info, device_));
    }
    void mutate(PopulationView offspring, const GenerationContext& context,
                cudaStream_t stream) override {
        auto guard = stream_guard(stream, device_);
        py::gil_scoped_acquire gil;
        auto output = object_.attr("mutate")(
            population_dict(offspring, device_), generation_context(context, device_));
        if (!output.is_none()) {
            auto source = checked_tensor(output, "mutated variables", device_, torch::kFloat32);
            if (source.sizes() != torch::IntArrayRef({offspring.size, offspring.dimension})) {
                throw std::invalid_argument("mutation output must have shape (population_size, dimension)");
            }
            view_float(offspring.variables, {offspring.size, offspring.dimension}, device_).copy_(source);
        }
    }
private:
    py::object object_;
    int device_ = 0;
};

class PythonReferences final : public IReferenceDirectionProvider {
public:
    explicit PythonReferences(py::object object) : object_(std::move(object)) {}
    void initialize(int requested, int objectives, CudaContext& cuda) override {
        device_ = current_device(); objectives_ = objectives;
        auto guard = stream_guard(cuda.execution_stream(), device_);
        py::gil_scoped_acquire gil;
        py::dict context; context["device"] = device_;
        auto output = object_.attr("initialize")(requested, objectives, context);
        set_directions(output, false);
        initial_ = directions_.clone();
    }
    ReferenceDirectionView view() noexcept override {
        return {directions_.defined() ? directions_.data_ptr<float>() : nullptr,
                initial_.defined() ? initial_.data_ptr<float>() : nullptr,
                nullptr,
                objectives_, count_};
    }
    void update(ConstPopulationView p, int active, const GenerationContext& context,
                cudaStream_t stream) override {
        py::gil_scoped_acquire gil;
        if (!py::hasattr(object_, "update")) return;
        auto guard = stream_guard(stream, device_);
        auto output = object_.attr("update")(
            population_dict(p, device_), directions_.transpose(0, 1), active,
            generation_context(context, device_));
        if (!output.is_none()) set_directions(output, true);
    }
    void reset(cudaStream_t stream) override {
        auto guard = stream_guard(stream, device_);
        directions_.copy_(initial_);
        py::gil_scoped_acquire gil;
        if (py::hasattr(object_, "reset")) object_.attr("reset")();
    }
private:
    void set_directions(py::handle value, bool fixed_shape) {
        auto source = checked_tensor(value, "reference directions", device_, torch::kFloat32);
        if (source.dim() != 2 || source.size(1) != objectives_) {
            throw std::invalid_argument("reference directions must have shape (count, objective_count)");
        }
        if (fixed_shape && source.size(0) != count_) {
            throw std::invalid_argument("updated reference directions cannot change count");
        }
        count_ = static_cast<int>(source.size(0));
        directions_ = source.transpose(0, 1).contiguous();
    }
    py::object object_;
    torch::Tensor directions_, initial_;
    int device_ = 0, objectives_ = 0, count_ = 0;
};

class PythonEnvironment final : public IEnvironmentSelector {
public:
    explicit PythonEnvironment(py::object object) : object_(std::move(object)) {}
    void initialize(const AlgorithmInfo& info, IReferenceDirectionProvider* refs,
                    CudaContext& cuda) override {
        device_ = current_device(); cuda_ = &cuda; size_ = info.population_size;
        auto guard = stream_guard(cuda.execution_stream(), device_);
        py::gil_scoped_acquire gil;
        py::dict info_dict; info_dict["name"] = info.name;
        info_dict["population_size"] = info.population_size;
        info_dict["dimension"] = info.dimension;
        info_dict["objective_count"] = info.objective_count;
        info_dict["max_generations"] = info.max_generations;
        py::object references = py::none();
        if (refs) {
            auto v = refs->view();
            if (v.values) references = py::cast(view_float(v.values,
                {v.objective_count, v.count}, device_).transpose(0, 1));
        }
        py::dict context; context["device"] = device_;
        object_.attr("initialize")(info_dict, references, context);
        active_ = size_;
    }
    void prepare(ConstPopulationView p, const GenerationContext& context) override {
        auto guard = stream_guard(context.cuda.execution_stream(), device_);
        py::gil_scoped_acquire gil;
        auto output = object_.attr("prepare")(
            population_dict(p, device_), generation_context(context, device_));
        update_state(output);
    }
    void select(ConstPopulationView parents, ConstPopulationView offspring,
                PopulationView next, const GenerationContext& context) override {
        auto guard = stream_guard(context.cuda.execution_stream(), device_);
        py::gil_scoped_acquire gil;
        auto output = object_.attr("select")(
            population_dict(parents, device_), population_dict(offspring, device_),
            generation_context(context, device_));
        if (!py::isinstance<py::dict>(output)) {
            throw std::invalid_argument("environment select must return a dict");
        }
        auto dict = py::reinterpret_borrow<py::dict>(output);
        if (dict.contains("indices")) {
            auto indices = checked_tensor(dict["indices"], "selected indices", device_, torch::kInt64);
            if (indices.numel() != next.size) throw std::invalid_argument("selected indices must contain population_size values");
            auto pv = population_dict(parents, device_);
            auto ov = population_dict(offspring, device_);
            auto variables = torch::cat({py::cast<torch::Tensor>(pv["variables"]), py::cast<torch::Tensor>(ov["variables"])}, 0).index_select(0, indices);
            auto objectives = torch::cat({py::cast<torch::Tensor>(pv["objectives"]), py::cast<torch::Tensor>(ov["objectives"])}, 0).index_select(0, indices);
            auto constraints = torch::cat({py::cast<torch::Tensor>(pv["constraints"]), py::cast<torch::Tensor>(ov["constraints"])}, 0).index_select(0, indices);
            view_float(next.variables, {next.size, next.dimension}, device_).copy_(variables);
            view_float(next.objectives, {next.objective_count, next.size}, device_).copy_(objectives.transpose(0, 1));
            view_float(next.constraints, {next.size}, device_).copy_(constraints);
        } else {
            auto variables = checked_tensor(dict["variables"], "variables", device_, torch::kFloat32);
            view_float(next.variables, {next.size, next.dimension}, device_).copy_(variables);
            copy_evaluation(dict, next, device_);
        }
        update_state(output);
    }
    MatingStateView mating_state() const noexcept override {
        return {
            rank_.defined() ? DeviceSpan<const int>{rank_.data_ptr<int>(), (size_t)rank_.numel()} : DeviceSpan<const int>{},
            reference_.defined() ? DeviceSpan<const int>{reference_.data_ptr<int>(), (size_t)reference_.numel()} : DeviceSpan<const int>{},
            score_.defined() ? DeviceSpan<const float>{score_.data_ptr<float>(), (size_t)score_.numel()} : DeviceSpan<const float>{},
            active_};
    }
    int active_count() const noexcept override { return active_; }
    DeviceSpan<const int> result_indices() const noexcept override {
        return result_.defined() ? DeviceSpan<const int>{result_.data_ptr<int>(), (size_t)result_.numel()} : DeviceSpan<const int>{};
    }
    void finalize(ConstPopulationView p, const GenerationContext& context) override {
        py::gil_scoped_acquire gil;
        if (!py::hasattr(object_, "finalize")) return;
        auto guard = stream_guard(context.cuda.execution_stream(), device_);
        update_state(object_.attr("finalize")(
            population_dict(p, device_), generation_context(context, device_)));
    }
    void reset() override {
        auto guard = stream_guard(cuda_->execution_stream(), device_);
        rank_ = torch::Tensor();
        reference_ = torch::Tensor();
        score_ = torch::Tensor();
        result_ = torch::Tensor();
        active_ = size_;
        py::gil_scoped_acquire gil;
        if (py::hasattr(object_, "reset")) object_.attr("reset")();
    }
private:
    void update_state(py::handle output) {
        if (!output || output.is_none() || !py::isinstance<py::dict>(output)) return;
        auto d = py::reinterpret_borrow<py::dict>(output);
        if (d.contains("rank")) rank_ = checked_tensor(d["rank"], "rank", device_, torch::kInt32);
        if (d.contains("reference_index")) reference_ = checked_tensor(d["reference_index"], "reference_index", device_, torch::kInt32);
        if (d.contains("score")) score_ = checked_tensor(d["score"], "score", device_, torch::kFloat32);
        if (d.contains("result_indices")) result_ = checked_tensor(d["result_indices"], "result_indices", device_, torch::kInt32);
        if (d.contains("active_count")) active_ = py::cast<int>(d["active_count"]);
    }
    py::object object_;
    CudaContext* cuda_ = nullptr;
    torch::Tensor rank_, reference_, score_, result_;
    int device_ = 0, size_ = 0, active_ = 0;
};

ProblemKind problem_kind(const std::string& name) {
    if (name == "DTLZ1") return ProblemKind::DTLZ1;
    if (name == "DTLZ2") return ProblemKind::DTLZ2;
    if (name == "DTLZ3") return ProblemKind::DTLZ3;
    if (name == "DTLZ4") return ProblemKind::DTLZ4;
    if (name == "DTLZ5") return ProblemKind::DTLZ5;
    if (name == "DTLZ6") return ProblemKind::DTLZ6;
    if (name == "DTLZ7") return ProblemKind::DTLZ7;
    if (name == "ConvexDTLZ2") return ProblemKind::ConvexDTLZ2;
    if (name == "C1DTLZ1") return ProblemKind::C1DTLZ1;
    if (name == "C1DTLZ3") return ProblemKind::C1DTLZ3;
    if (name == "C2DTLZ2") return ProblemKind::C2DTLZ2;
    if (name == "C2ConvexDTLZ2") return ProblemKind::C2ConvexDTLZ2;
    if (name == "C3DTLZ1") return ProblemKind::C3DTLZ1;
    if (name == "C3DTLZ4") return ProblemKind::C3DTLZ4;
    if (name == "CSDP") return ProblemKind::CSDP;
    throw std::invalid_argument("unknown problem kind: " + name);
}

} // namespace

std::unique_ptr<IProblemEvaluator> make_problem(const py::dict& s) {
    if (type_of(s) == "NativeProblem") return make_native_problem(s);
    if (type_of(s) == "PythonProblem") return std::make_unique<PythonProblem>(s);
    return ProblemFactory::create(problem_kind(py::cast<std::string>(s["kind"])),
        py::cast<int>(s["dimension"]), py::cast<int>(s["objectives"]),
        py::cast<float>(s["constraint_activation_ratio"]));
}

py::dict benchmark_problem(const py::dict& spec, py::object variables, int repeats, int warmup) {
    if (repeats <= 0 || warmup < 0) throw std::invalid_argument("Invalid benchmark repeat/warmup count");
    auto x = py::cast<torch::Tensor>(variables);
    if (!x.is_cuda() || x.scalar_type() != torch::kFloat32 || x.dim() != 2 ||
        x.size(0) <= 0 || x.size(0) > INT_MAX || x.size(1) > INT_MAX)
        throw std::invalid_argument("variables must be a nonempty 2-D float32 CUDA Tensor");
    c10::cuda::CUDAGuard device_guard(x.device());
    x = x.contiguous();
    CudaConfig config;
    config.device_id = x.get_device(); config.enable_warmup = false;
    CudaContext cuda(config);
    // Make input production visible before using the independent evaluation stream.
    auto check = [](cudaError_t status) {
        if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    };
    check(cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream(x.get_device()).stream()));
    auto guard = stream_guard(cuda.evaluation_stream(), x.get_device());
    auto problem = make_problem(spec);
    const auto& info = problem->info();
    if (x.size(1) != info.dimension) throw std::invalid_argument("variables dimension does not match problem");
    Population population(x.size(0), info.dimension, info.objective_count, cuda);
    // Bounds are allocated on the execution stream, then written below on the
    // evaluation stream. Honor the async allocation's stream ordering.
    cuda.wait_execution_on_evaluation();
    population.set_bounds(info.lower_bounds, info.upper_bounds, cuda.evaluation_stream());
    auto view = population.view();
    view.variables = x.data_ptr<float>();
    struct Event {
        cudaEvent_t value = nullptr;
        ~Event() { if (value) cudaEventDestroy(value); }
    } start, stop;
    check(cudaEventCreate(&start.value)); check(cudaEventCreate(&stop.value));
    double wall_ms;
    float cuda_ms;
    {
        py::gil_scoped_release release;
        problem->initialize(cuda);
        const EvaluationContext context{0, 1, cuda};
        for (int i = 0; i < warmup; ++i) problem->evaluate(view, context, cuda.evaluation_stream());
        cuda.synchronize();
        auto begin = std::chrono::steady_clock::now();
        check(cudaEventRecord(start.value, cuda.evaluation_stream()));
        for (int i = 0; i < repeats; ++i) problem->evaluate(view, context, cuda.evaluation_stream());
        check(cudaEventRecord(stop.value, cuda.evaluation_stream()));
        check(cudaEventSynchronize(stop.value));
        wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
        check(cudaEventElapsedTime(&cuda_ms, start.value, stop.value));
    }
    py::dict result;
    result["wall_ms_per_eval"] = wall_ms / repeats;
    result["cuda_ms_per_eval"] = cuda_ms / repeats;
    return result;
}

std::unique_ptr<IMatingSelector> make_mating(const py::dict& s) {
    const auto t = type_of(s);
    if (t == "PythonMating") return std::make_unique<PythonMating>(s["object"]);
    return OperatorFactory::mating(t == "RandomMating" ? MatingKind::Random : MatingKind::Tournament);
}

std::unique_ptr<ICrossoverOperator> make_crossover(const py::dict& s) {
    if (type_of(s) == "PythonCrossover") return std::make_unique<PythonCrossover>(s["object"]);
    SBXConfig c{
        py::cast<float>(s["eta_initial"]),
        py::cast<float>(s["eta_final"]),
        py::cast<float>(s["probability"]),
        py::cast<float>(s["variable_copy_probability"]),
    };
    return OperatorFactory::crossover(CrossoverKind::SBX, c);
}

std::unique_ptr<IMutationOperator> make_mutation(const py::dict& s) {
    const auto t = type_of(s);
    if (t == "PythonMutation") return std::make_unique<PythonMutation>(s["object"]);
    if (t == "NoMutation") return OperatorFactory::mutation(MutationKind::None);
    PolynomialMutationConfig c{py::cast<float>(s["eta_initial"]), py::cast<float>(s["eta_final"]), py::cast<float>(s["probability"])};
    return OperatorFactory::mutation(MutationKind::Polynomial, c);
}

std::unique_ptr<IReferenceDirectionProvider> make_references(const py::dict& s) {
    const auto t = type_of(s);
    if (t == "PythonReferenceDirections") return std::make_unique<PythonReferences>(s["object"]);
    if (t == "DasDennisDirections") return std::make_unique<DasDennisDirections>(DasDennisConfig{py::cast<int>(s["partitions"])});
    if (t == "AdaptiveRVEADirections") return std::make_unique<AdaptiveRVEADirections>(AdaptiveDirectionConfig{py::cast<float>(s["frequency"])});
    auto tensor = py::cast<torch::Tensor>(s["values"]).to(torch::kCPU).to(torch::kFloat32).contiguous();
    if (tensor.dim() != 2) throw std::invalid_argument("user directions must be a 2-D tensor");
    auto objective_major = tensor.transpose(0, 1).contiguous();
    std::vector<float> values(objective_major.numel());
    std::copy(objective_major.data_ptr<float>(), objective_major.data_ptr<float>() + objective_major.numel(), values.begin());
    return std::make_unique<UserDefinedDirections>(std::move(values), tensor.size(1), py::cast<bool>(s["normalize"]));
}

std::unique_ptr<IEnvironmentSelector> make_environment(const py::dict& s) {
    const auto t = type_of(s);
    if (t == "PythonEnvironmentSelector") return std::make_unique<PythonEnvironment>(s["object"]);
    if (t == "RVEAEnvironmentSelector") return std::make_unique<RVEAEnvironmentSelector>(RVEASelectionConfig{py::cast<float>(s["alpha"])});
    NSGA3SelectionConfig c;
    c.sparse_ratio = py::cast<float>(s["sparse_ratio"]);
    c.cv_quantization = {py::cast<int>(s["cv_bins"]), py::cast<float>(s["cv_clip_upper"]),
        py::cast<float>(s["cv_log_alpha"]), py::cast<float>(s["feasibility_epsilon"])};
    return std::make_unique<NSGA3EnvironmentSelector>(c);
}

} // namespace cuda_moea::python
