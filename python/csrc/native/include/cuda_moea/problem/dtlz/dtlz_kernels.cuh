#pragma once

#include "cuda_moea/core/cuda/cuda_manager.cuh"
#include "evaluation_workspace.cuh"
#include "population_adapter.cuh"

// ======================================================================================================================================================= //
//                                              DTLZ Compute Functions                                                                                       //
// ======================================================================================================================================================= //

// Unconstrained DTLZ
void compute_dtlz1(float* d_pop, float* d_fit_trans, float* d_fit, int N, int D, int M, cudaStream_t fit_stream);
void compute_dtlz2(float* d_pop, float* d_fit_trans, float* d_fit, int N, int D, int M, cudaStream_t fit_stream);
void compute_dtlz3(float* d_pop, float* d_fit_trans, float* d_fit, int N, int D, int M, cudaStream_t fit_stream);
void compute_dtlz4(float* d_pop, float* d_fit_trans, float* d_fit, int N, int D, int M, cudaStream_t fit_stream);
void compute_dtlz5(float* d_pop, float* d_fit_trans, float* d_fit, int N, int D, int M, cudaStream_t fit_stream);
void compute_dtlz6(float* d_pop, float* d_fit_trans, float* d_fit, int N, int D, int M, cudaStream_t fit_stream);
void compute_dtlz7(float* d_pop, float* d_fit_trans, float* d_fit, int N, int D, int M, cudaStream_t fit_stream);

// Constrained DTLZ
void compute_c1_dtlz1(float* d_pop, float* d_fit_trans, float* d_fit, float* d_cv, int N, int D, int M, cudaStream_t fit_stream);
void compute_c1_dtlz3(float* d_pop, float* d_fit_trans, float* d_fit, float* d_cv, int N, int D, int M, cudaStream_t fit_stream);
void compute_c2_dtlz2(float* d_pop, float* d_fit_trans, float* d_fit, float* d_cv, int N, int D, int M, cudaStream_t fit_stream);
void compute_c2_convex_dtlz2(float* d_pop, float* d_fit_trans, float* d_fit, float* d_cv, int N, int D, int M, cudaStream_t fit_stream);
void compute_c3_dtlz1(float* d_pop, float* d_fit_trans, float* d_fit, float* d_cv, int N, int D, int M, cudaStream_t fit_stream);
void compute_c3_dtlz4(float* d_pop, float* d_fit_trans, float* d_fit, float* d_cv, int N, int D, int M, cudaStream_t fit_stream);
void compute_csdp(float* d_pop, float* d_fit_trans, float* d_fit, float* d_cv, int N, int D, int M, cudaStream_t fit_stream);

// ======================================================================================================================================================= //
//                                              MOEAStdTestEvaluator                                                                                         //
// ======================================================================================================================================================= //
#include "evaluator_adapter_base.cuh"

struct MOEAStdTestEvaluator : public IMOPEvaluator {
    struct Config {
        float cv_activation_ratio;

        Config() : cv_activation_ratio(0.0f) {}
        explicit Config(float ratio) : cv_activation_ratio(ratio) {}
    };

    MOPAuxData& d_auxdata;
    int M, N;
    bool enable_h_save;
    Config config;
    int n_iter = -1;
    int N_iter = 1;
    int last_prepared_target_iter = -1;

    using IMOPEvaluator::prepare_parent_cv;

    MOEAStdTestEvaluator(MOPAuxData& auxdata, int _M, int _N, bool save)
        : d_auxdata(auxdata), M(_M), N(_N), enable_h_save(save), config() {}

    MOEAStdTestEvaluator(MOPAuxData& auxdata, int _M, int _N, bool save, Config cfg)
        : d_auxdata(auxdata), M(_M), N(_N), enable_h_save(save), config(cfg) {}

    void set_iteration_context(int _n_iter, int _N_iter) override;
    bool is_cv_active(int eval_iter) const;
    bool is_cv_activation_switch_node(int target_iter) const;
    void evaluate(cudaStreamSync, const PopData&, float*, float*) override;
    bool prepare_parent_cv(cudaStreamSync, const PopData&, const float*, float*, int, int) override;
};
