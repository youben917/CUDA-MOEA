#pragma once

namespace cuda_moea {

class CudaContext;

void warmup_cuda_libraries(CudaContext& context);

} // namespace cuda_moea
