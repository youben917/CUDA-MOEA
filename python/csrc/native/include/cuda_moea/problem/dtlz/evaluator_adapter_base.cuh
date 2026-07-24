#pragma once

#include "cuda_moea/core/cuda/cuda_manager.cuh"

struct PopData;  // forward declare
struct CongestionWorkloadStats;

struct IMOPEvaluator {
    virtual ~IMOPEvaluator() = default;
    virtual void evaluate(
        cudaStreamSync cuda_streams,
        const PopData& d_pop,
        float* d_cv,
        float* d_fv) = 0;
    // Default no-op for evaluators without iteration-dependent behavior.
    virtual void set_iteration_context(int /*n_iter*/, int /*N_iter*/) {}
    // Optional hook: prepare parent CV semantics for a target iteration before mating/merge.
    virtual bool prepare_parent_cv(
        cudaStreamSync /*cuda_streams*/,
        const PopData& /*d_parent*/,
        float* /*d_cv*/,
        int /*target_iter*/,
        int /*n_active*/) { return false; }
    virtual bool prepare_parent_cv(
        cudaStreamSync cuda_streams,
        const PopData& d_parent,
        const float* /*d_fv*/,
        float* d_cv,
        int target_iter,
        int n_active) {
        return prepare_parent_cv(cuda_streams, d_parent, d_cv, target_iter, n_active);
    }
    // Optional hook: provide NSGA-III CV quantization params to evaluators that
    // implement CV-side semantics coupled to ndsort quantization (default no-op).
    virtual void configure_cv_quant_for_absorption(
        int /*cv_quant_bins*/,
        float /*cv_quant_clip_upper*/,
        float /*cv_quant_log_alpha*/,
        float /*cv_quant_eps_feas*/) {}
    // Optional hook: request auto-clip calculation on a parent-refresh node.
    virtual bool should_auto_clip_on_refresh(int /*target_iter*/) const { return false; }
    // Optional hook: provide a per-evaluation congestion workload accumulator.
    virtual void set_congestion_workload_stats(CongestionWorkloadStats* /*stats*/) {}

    IMOPEvaluator() = default;
    IMOPEvaluator(const IMOPEvaluator&) = delete;
    IMOPEvaluator& operator=(const IMOPEvaluator&) = delete;
    IMOPEvaluator(IMOPEvaluator&&) = default;
    IMOPEvaluator& operator=(IMOPEvaluator&&) = default;
};
