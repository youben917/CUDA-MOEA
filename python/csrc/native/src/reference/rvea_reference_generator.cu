#include "cuda_moea/reference/rvea_reference_generator.cuh"

#include <algorithm>
#include <cmath>
#include <utility>

// ==================================================================================== //
// Helper functions
namespace {

    // Computes the binomial coefficient C(n, k), i.e., "n choose k"
    long long comb(int n, int k) {
        if (k > n) return 0;
        if (k == 0 || k == n) return 1;
        k = std::min(k, n - k);
        long long result = 1;
        for (int i = 1; i <= k; ++i) {
            result *= (n - k + i);
            result /= i;
        }
        return result;
    }

    int compute_n_ref_raw(int N_approx, int M) {
        int h1 = 0;
        while (comb(h1 + M - 1, M - 1) <= N_approx) {
            ++h1;
        }
        --h1;

        long long N_layer1 = comb(h1 + M - 1, M - 1);
        long long N_layer2 = 0;
        if (h1 < M) {
            int h2 = 1;
            while (N_layer1 + comb(h2 + M - 1, M - 1) <= N_approx) {
                ++h2;
            }
            --h2;
            if (h2 > 0) {
                N_layer2 = comb(h2 + M - 1, M - 1);
            }
        }
        return static_cast<int>(N_layer1 + N_layer2);
    }

    int compute_even_n_ref_from_approx(int N_approx, int M) {
        int N_ref = compute_n_ref_raw(N_approx, M);
        if ((N_ref & 1) == 0) {
            return N_ref;
        }

        // First retry from N_approx + 1, then keep minimal upward search until even.
        int adjusted_N = N_approx + 1;
        do {
            N_ref = compute_n_ref_raw(adjusted_N, M);
            ++adjusted_N;
        } while (N_ref & 1);

        return N_ref;
    }

    // Internal helper: generates reference points for a single layer
    // Returns: pair of (transposed_points, num_points)
    // transposed_points: vector in row-major (M, num_points) layout
    std::pair<std::vector<float>, int> generate_single_layer(int h, int M) {
        long long num_points = comb(h + M - 1, M - 1);
        int N_layer = static_cast<int>(num_points);
        
        std::vector<bool> mask(h + M - 1);
        std::fill(mask.begin(), mask.begin() + (M - 1), true);
        
        // Pre-allocate output in transposed layout: (M, N_layer) row-major
        std::vector<float> transposed(M * N_layer, 0.0f);
        
        int point_idx = 0;
        
        // Generate reference points using combinatorial approach
        do {
            // Build full combination vector with sentinel boundaries
            std::vector<int> combo;
            combo.reserve(M + 1);
            combo.push_back(-1);
            for (int i = 0; i < (int)mask.size(); ++i) {
                if (mask[i]) combo.push_back(i);
            }
            combo.push_back(h + M - 1);
            
            // Compute and store coordinates for this reference point
            // Store in column point_idx
            for (int dim_idx = 0; dim_idx < M; ++dim_idx) {
                float val = (combo[dim_idx+1] - combo[dim_idx] - 1) / static_cast<float>(h);
                transposed[dim_idx * N_layer + point_idx] = val;
            }
            
            ++point_idx;
        } while (std::prev_permutation(mask.begin(), mask.end()));
        
        return {std::move(transposed), N_layer};
    }

    // Generates reference points using Das & Dennis method with two-layer approach
    // Returns: pair of (transposed_points, N_ref)
    std::pair<std::vector<float>, int> reference_points(int N, int M) {
        // Step 1: Determine the largest h1 such that C(h1 + M - 1, M - 1) <= N
        int h1 = 0;
        while (comb(h1 + M - 1, M - 1) <= N) {
            ++h1;
        }
        --h1;  // ensure not exceeding N
        
        // Step 2: Generate primary layer points
        auto [layer1_points, N_layer1] = generate_single_layer(h1, M);
        
        // Step 3: Determine if secondary layer is needed and feasible
        int h2 = 0;
        std::vector<float> layer2_points;
        int N_layer2 = 0;
        
        if (h1 < M) {
            // Find largest h2 such that combined points don't exceed N
            h2 = 1;
            while (comb(h1 + M - 1, M - 1) + comb(h2 + M - 1, M - 1) <= N) {
                ++h2;
            }
            --h2;
            
            if (h2 > 0) {
                // Generate secondary layer points
                auto [temp_points, temp_N] = generate_single_layer(h2, M);
                N_layer2 = temp_N;
                layer2_points.resize(M * N_layer2);
                
                // Transform secondary layer: scale by 0.5 and shift toward centroid
                // Formula: 0.5 * (points + 1/M)
                float centroid_val = 1.0f / M;
                for (int point_idx = 0; point_idx < N_layer2; ++point_idx) {
                    for (int dim_idx = 0; dim_idx < M; ++dim_idx) {
                        float val = temp_points[dim_idx * N_layer2 + point_idx];
                        layer2_points[dim_idx * N_layer2 + point_idx] = 0.5f * (val + centroid_val);
                    }
                }
            }
        }
        
        // Step 4: Combine both layers into final output
        int N_ref = N_layer1 + N_layer2;
        std::vector<float> transposed(M * N_ref, 0.0f);
        
        // Copy primary layer
        for (int point_idx = 0; point_idx < N_layer1; ++point_idx) {
            for (int dim_idx = 0; dim_idx < M; ++dim_idx) {
                transposed[dim_idx * N_ref + point_idx] = layer1_points[dim_idx * N_layer1 + point_idx];
            }
        }
        
        // Copy secondary layer (if exists)
        if (N_layer2 > 0) {
            for (int point_idx = 0; point_idx < N_layer2; ++point_idx) {
                for (int dim_idx = 0; dim_idx < M; ++dim_idx) {
                    transposed[dim_idx * N_ref + N_layer1 + point_idx] =
                        layer2_points[dim_idx * N_layer2 + point_idx];
                }
            }
        }
        
        return {std::move(transposed), N_ref};
    }

} // end anonymous namespace

int rvea::compute_N_ref(int N_approx, int M) {
    return compute_even_n_ref_from_approx(N_approx, M);
}

// ==================================================================================== //
// Generate reference points on host side with normalization
// Returns: tuple of (h_rps_trans, h_nrps_trans, N_ref)
// h_rps_trans:  (M, N_ref) - reference points with zero padding, row-major
// h_nrps_trans: (M, N_ref) - normalized reference points with zero padding, row-major
// N_ref: actual number of reference points generated (including both layers)
// Layout: row-major (M, N_ref), where each column represents one reference point
std::tuple<std::vector<float>, std::vector<float>, int> rvea::initialize_refpts(
    int N,   // input: scalar - approximate number of desired points
    int M    // input: scalar - number of objectives
) {
    // Step 1: Generate reference points in transposed layout (M, N_ref)
    // This now includes both primary and secondary layers
    auto [h_rps_trans, N_ref] = reference_points(N, M);

    // Match EvoX uniform_sampling: boundary coordinates are lifted before
    // cosine-based association so that no reference direction contains zeros.
    for (float& value : h_rps_trans) {
        value = std::max(value, 1e-6f);
    }
    
    // Step 2: Allocate normalized array in fp32 for computation
    std::vector<float> h_nrps_trans_fp32(M * N_ref, 0.0f);
    
    // Step 3: Compute normalized reference points in fp32 (high precision)
    for (int point_idx = 0; point_idx < N_ref; ++point_idx) {
        // Calculate L2 norm in fp32
        float sum_sq = 0.0f;
        for (int dim_idx = 0; dim_idx < M; ++dim_idx) {
            float val = h_rps_trans[dim_idx * N_ref + point_idx];
            sum_sq += val * val;
        }
        float norm = sqrtf(sum_sq);
        
        // Normalize in fp32
        for (int dim_idx = 0; dim_idx < M; ++dim_idx) {
            h_nrps_trans_fp32[dim_idx * N_ref + point_idx] = 
                h_rps_trans[dim_idx * N_ref + point_idx] / norm;
        }
    }
    
    return std::make_tuple(std::move(h_rps_trans), std::move(h_nrps_trans_fp32), N_ref);
}
