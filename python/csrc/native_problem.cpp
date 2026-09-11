#include "native_problem.h"
#include "cuda_moea/native/plugin.cuh"

#include <cmath>
#include <cstring>
#include <dlfcn.h>
#include <set>
#include <stdexcept>
#include <pybind11/stl.h>

namespace cuda_moea::python {
namespace {
struct Module {
    void* handle = nullptr;
    const native::Plugin* plugin = nullptr;
    explicit Module(const std::string& path) {
        handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!handle) throw std::runtime_error(std::string("Cannot load native problem: ") + dlerror());
        try {
            auto entry = reinterpret_cast<native::Entry>(dlsym(handle, "cuda_moea_problem_v1"));
            if (!entry) throw std::invalid_argument("Missing cuda_moea_problem_v1 registration");
            plugin = entry();
            if (!plugin || plugin->abi != 1 || !plugin->build_id ||
                std::strcmp(plugin->build_id, CUDA_MOEA_NATIVE_BUILD_ID) != 0) {
                throw std::invalid_argument("Native problem SDK mismatch; rebuild with this cuda_moea installation");
            }
            if (!plugin->name || !plugin->parameters || !plugin->create || !plugin->destroy)
                throw std::invalid_argument("Incomplete native problem registration");
            std::set<std::string> names;
            for (const auto& parameter : *plugin->parameters) {
                if (!parameter.name || !names.insert(parameter.name).second)
                    throw std::invalid_argument("Invalid or duplicate native parameter name");
            }
        } catch (...) { dlclose(handle); handle = nullptr; throw; }
    }
    ~Module() { if (handle) dlclose(handle); }
    Module(const Module&) = delete;
    Module& operator=(const Module&) = delete;
};

native::Parameter parameter_value(py::handle value, const native::ParameterSpec& spec) {
    auto fail = [&]() -> native::Parameter {
        throw std::invalid_argument(std::string("Invalid type/value for native parameter '") + spec.name + "'");
    };
    switch (spec.default_value.index()) {
    case 0:
        if (!py::isinstance<py::int_>(value) || py::isinstance<py::bool_>(value)) return fail();
        return py::cast<std::int64_t>(value);
    case 1: {
        if ((!py::isinstance<py::float_>(value) && !py::isinstance<py::int_>(value)) || py::isinstance<py::bool_>(value)) return fail();
        double result = py::cast<double>(value);
        if (!std::isfinite(result)) return fail();
        return result;
    }
    case 2:
        if (!py::isinstance<py::bool_>(value)) return fail();
        return py::cast<bool>(value);
    case 3:
        if (!py::isinstance<py::str>(value)) return fail();
        return py::cast<std::string>(value);
    case 4: {
        if (!py::isinstance<py::list>(value) && !py::isinstance<py::tuple>(value)) return fail();
        std::vector<double> result;
        native::ParameterSpec element{spec.name, 0.0, ""};
        for (auto item : py::reinterpret_borrow<py::sequence>(value))
            result.push_back(std::get<double>(parameter_value(item, element)));
        return result;
    }
    }
    return fail();
}

class NativeProblem final : public IProblemEvaluator {
public:
    explicit NativeProblem(const py::dict& spec)
        : module_(std::make_unique<Module>(py::cast<std::string>(spec["library"]))) {
        const auto& plugin = *module_->plugin;
        if (py::cast<std::string>(spec["name"]) != plugin.name)
            throw std::invalid_argument("Native problem name does not match registration");
        native::ProblemConfig config{
            py::cast<int>(spec["dimension"]), py::cast<int>(spec["objectives"]),
            py::cast<std::vector<float>>(spec["lower_bounds"]),
            py::cast<std::vector<float>>(spec["upper_bounds"]), {}};
        auto parameters = py::cast<py::dict>(spec["parameters"]);
        for (const auto& item : *plugin.parameters) {
            config.parameters.emplace(item.name, parameters.contains(item.name)
                ? parameter_value(parameters[item.name], item) : item.default_value);
        }
        for (auto item : parameters) {
            if (!config.parameters.count(py::cast<std::string>(item.first)))
                throw std::invalid_argument("Unknown native problem parameter: " + py::cast<std::string>(item.first));
        }
        try {
            instance_ = plugin.create(config);
        } catch (const std::exception& error) {
            // Do not propagate a plugin-defined exception type past dlclose.
            throw std::invalid_argument(std::string("Native problem construction failed: ") + error.what());
        } catch (...) {
            throw std::runtime_error("Native problem construction failed with a nonstandard exception");
        }
        if (!instance_) throw std::runtime_error("Native problem factory returned null");
        const auto& info = instance_->info();
        if (info.dimension != config.dimension || info.objective_count != config.objectives ||
            info.lower_bounds != config.lower_bounds || info.upper_bounds != config.upper_bounds ||
            info.constraint_count < 0) {
            plugin.destroy(instance_); instance_ = nullptr;
            throw std::invalid_argument("Native ProblemInfo must match the requested dimensions and bounds");
        }
    }
    ~NativeProblem() override {
        int previous = -1;
        if (device_ >= 0) {
            cudaGetDevice(&previous);
            cudaSetDevice(device_);
            cudaStreamSynchronize(evaluation_);
            cudaStreamSynchronize(execution_);
        }
        if (instance_) module_->plugin->destroy(instance_);
        // A destructor may enqueue workspace frees. Finish before dlclose.
        if (device_ >= 0) {
            cudaStreamSynchronize(evaluation_);
            cudaStreamSynchronize(execution_);
        }
        module_.reset();
        if (previous >= 0) cudaSetDevice(previous);
    }
    const ProblemInfo& info() const noexcept override { return instance_->info(); }
    void initialize(CudaContext& cuda) override {
        auto status = cudaGetDevice(&device_);
        if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
        evaluation_ = cuda.evaluation_stream(); execution_ = cuda.execution_stream();
        instance_->initialize(cuda);
    }
    void evaluate(PopulationView p, const EvaluationContext& c, cudaStream_t s) override {
        instance_->evaluate(p, c, s);
        auto status = cudaGetLastError();
        if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    }
    bool prepare_parent(PopulationView p, const EvaluationContext& c, cudaStream_t s) override {
        return instance_->prepare_parent(p, c, s);
    }
    void reset() override { instance_->reset(); }
private:
    std::unique_ptr<Module> module_;
    IProblemEvaluator* instance_ = nullptr;
    int device_ = -1;
    cudaStream_t evaluation_ = nullptr, execution_ = nullptr;
};
} // namespace

std::unique_ptr<IProblemEvaluator> make_native_problem(const py::dict& spec) {
    return std::make_unique<NativeProblem>(spec);
}

py::dict native_problem_schema(const std::string& path) {
    Module module(path);
    py::dict result, parameters;
    const char* types[] = {"int", "float", "bool", "str", "float_list"};
    result["name"] = module.plugin->name;
    result["build_id"] = module.plugin->build_id;
    for (const auto& item : *module.plugin->parameters) {
        py::dict p;
        p["type"] = types[item.default_value.index()];
        p["default"] = std::visit([](const auto& value) { return py::cast(value); }, item.default_value);
        p["description"] = item.description ? item.description : "";
        parameters[item.name] = p;
    }
    result["parameters"] = parameters;
    return result;
}
} // namespace cuda_moea::python
