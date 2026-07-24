#include <memory>
#include <stdexcept>
#include <string>

#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "adapters.h"
#include "cuda_moea/factory/algorithm_factory.cuh"

namespace py = pybind11;
using namespace cuda_moea;

namespace {

class PyAlgorithm : public std::enable_shared_from_this<PyAlgorithm> {
public:
    PyAlgorithm(const std::string& kind, const py::dict& config,
                const py::dict& problem, const py::dict& mating,
                const py::dict& crossover, const py::dict& mutation,
                const py::dict& references, const py::dict& environment)
        : algorithm_(build(kind, config, problem, mating, crossover,
                           mutation, references, environment)) {}

    void initialize() { py::gil_scoped_release release; algorithm_.initialize(); }
    void step() { py::gil_scoped_release release; algorithm_.step(); }
    void reset() { py::gil_scoped_release release; algorithm_.reset(); }
    void synchronize() { py::gil_scoped_release release; algorithm_.synchronize(); }

    py::dict run(bool copy) {
        DeviceRunResult value;
        { py::gil_scoped_release release; value = algorithm_.run_device(); }
        return make_result(value, copy, true);
    }

    py::dict result(bool copy) {
        DeviceRunResult value;
        { py::gil_scoped_release release; value = algorithm_.device_result(); }
        return make_result(value, copy, true);
    }

    py::dict population(bool copy) {
        if (!algorithm_.initialized()) throw std::logic_error("Algorithm has not been initialized");
        algorithm_.synchronize();
        auto p = algorithm_.population().view();
        DeviceRunResult value;
        value.population = p;
        int device = 0;
        cudaGetDevice(&device);
        value.device_id = device;
        return make_result(value, copy, false);
    }

    bool initialized() const noexcept { return algorithm_.initialized(); }
    bool finished() const noexcept { return algorithm_.finished(); }
    int generation() const noexcept { return algorithm_.generation(); }

private:
    static Algorithm build(const std::string& name, const py::dict& config,
                           const py::dict& problem, const py::dict& mating,
                           const py::dict& crossover, const py::dict& mutation,
                           const py::dict& references, const py::dict& environment) {
        const auto kind = name == "RVEA" ? AlgorithmKind::RVEA : AlgorithmKind::NSGA3;
        auto state = AlgorithmFactory::defaults(kind);
        state.config.population_size = py::cast<int>(config["population_size"]);
        state.config.max_generations = py::cast<int>(config["max_generations"]);
        state.config.progress_interval = py::cast<int>(config["progress_interval"]);
        state.config.print_progress = py::cast<bool>(config["print_progress"]);
        const auto cuda = py::cast<py::dict>(config["cuda"]);
        state.config.cuda.device_id = py::cast<int>(cuda["device_id"]);
        state.config.cuda.evaluation_pool_ratio = py::cast<float>(cuda["evaluation_pool_ratio"]);
        state.config.cuda.execution_pool_ratio = py::cast<float>(cuda["execution_pool_ratio"]);
        state.config.cuda.memory_pool_policy = py::cast<int>(cuda["memory_pool_policy"]);
        state.config.cuda.seed = py::cast<unsigned long long>(cuda["seed"]);
        state.config.cuda.enable_warmup = py::cast<bool>(cuda["enable_warmup"]);
        state.config.data_save.enabled = py::cast<bool>(config["save_enabled"]);
        state.config.data_save.output_directory = py::cast<std::string>(config["save_directory"]);
        state.config.data_save.generation_interval = py::cast<int>(config["save_interval"]);
        if (!config["initial_population"].is_none()) {
            auto tensor = py::cast<torch::Tensor>(config["initial_population"])
                .detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
            if (tensor.dim() != 2 || tensor.size(0) != state.config.population_size) {
                throw std::invalid_argument(
                    "initial_population must have shape (population_size, dimension)");
            }
            state.config.initial_population.assign(
                tensor.data_ptr<float>(),
                tensor.data_ptr<float>() + tensor.numel());
        }
        state.strategies.problem = cuda_moea::python::make_problem(problem);
        state.strategies.mating = cuda_moea::python::make_mating(mating);
        state.strategies.crossover = cuda_moea::python::make_crossover(crossover);
        state.strategies.mutation = cuda_moea::python::make_mutation(mutation);
        state.strategies.reference_directions = cuda_moea::python::make_references(references);
        state.strategies.environment_selector = cuda_moea::python::make_environment(environment);
        return AlgorithmFactory::create(std::move(state));
    }

    torch::Tensor float_view(const float* pointer, std::vector<int64_t> shape,
                             int device) {
        auto owner = shared_from_this();
        return torch::from_blob(const_cast<float*>(pointer), std::move(shape),
            [owner = std::move(owner)](void*) mutable { owner.reset(); },
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, device));
    }

    torch::Tensor int_view(const int* pointer, std::vector<int64_t> shape,
                           int device) {
        auto owner = shared_from_this();
        return torch::from_blob(const_cast<int*>(pointer), std::move(shape),
            [owner = std::move(owner)](void*) mutable { owner.reset(); },
            torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, device));
    }

    py::dict make_result(const DeviceRunResult& value, bool copy, bool metadata) {
        const auto& p = value.population;
        auto variables = float_view(p.variables, {p.size, p.dimension}, value.device_id);
        auto objectives = float_view(p.objectives, {p.objective_count, p.size},
                                     value.device_id).transpose(0, 1);
        auto constraints = float_view(p.constraints, {p.size}, value.device_id);
        if (copy) {
            variables = variables.clone();
            objectives = objectives.contiguous();
            constraints = constraints.clone();
        }
        py::dict result;
        result["variables"] = variables;
        result["objectives"] = objectives;
        result["constraints"] = constraints;
        if (metadata) {
            torch::Tensor auxiliary;
            if (!value.auxiliary_indices.empty()) {
                auxiliary = int_view(value.auxiliary_indices.data,
                                     {(long)value.auxiliary_indices.size}, value.device_id);
                if (copy) auxiliary = auxiliary.clone();
            } else {
                auxiliary = torch::empty({0}, torch::TensorOptions()
                    .dtype(torch::kInt32).device(torch::kCUDA, value.device_id));
            }
            result["auxiliary_indices"] = auxiliary;
            result["active_count"] = value.active_count;
            result["total_ms"] = value.total_ms;
        }
        return result;
    }

    Algorithm algorithm_;
};

} // namespace

PYBIND11_MODULE(_C, module) {
    module.doc() = "PyTorch bindings for CUDA-MOEA";
    py::class_<PyAlgorithm, std::shared_ptr<PyAlgorithm>>(module, "Algorithm")
        .def(py::init<const std::string&, const py::dict&, const py::dict&,
                      const py::dict&, const py::dict&, const py::dict&,
                      const py::dict&, const py::dict&>())
        .def("initialize", &PyAlgorithm::initialize)
        .def("step", &PyAlgorithm::step)
        .def("run", &PyAlgorithm::run, py::arg("copy") = true)
        .def("result", &PyAlgorithm::result, py::arg("copy") = true)
        .def("population", &PyAlgorithm::population, py::arg("copy") = false)
        .def("reset", &PyAlgorithm::reset)
        .def("synchronize", &PyAlgorithm::synchronize)
        .def_property_readonly("initialized", &PyAlgorithm::initialized)
        .def_property_readonly("finished", &PyAlgorithm::finished)
        .def_property_readonly("generation", &PyAlgorithm::generation);
}
