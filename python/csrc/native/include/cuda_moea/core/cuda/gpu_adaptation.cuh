#pragma once

#ifndef GPU_CONSTANTS_CUH
#define GPU_CONSTANTS_CUH
namespace gpu {
    namespace models {
        namespace rtx4060 {
            constexpr int SM_COUNT           = 24;
            constexpr int MAX_THREADS_PER_SM = 1536;
            constexpr int MAX_THREADS        = 36864;  // 24 * 1536
        }   
        namespace rtx4090 {
            constexpr int SM_COUNT           = 128;
            constexpr int MAX_THREADS_PER_SM = 1536;
            constexpr int MAX_THREADS        = 196608;  // 128 * 1536
        }
        namespace rtx5090 {
            constexpr int SM_COUNT           = 170;
            constexpr int MAX_THREADS_PER_SM = 1536;
            constexpr int MAX_THREADS        = 261120;  // 170 * 1536 
        }
        namespace rtx6000pro {
            constexpr int SM_COUNT           = 188;
            constexpr int MAX_THREADS_PER_SM = 1536;
            constexpr int MAX_THREADS        = 288768;  // 188 * 1536 
        }
        namespace h800 {
            constexpr int SM_COUNT           = 132;
            constexpr int MAX_THREADS_PER_SM = 2048;
            constexpr int MAX_THREADS        = 270336;  // 270,336
        }

    }
}
#endif