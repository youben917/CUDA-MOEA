#pragma once

__device__ __forceinline__ float min_vec(const float2& v) {
    return fminf(v.x, v.y);
}

__device__ __forceinline__ float min_vec(const float4& v) {
    return fminf(fminf(v.x, v.y), fminf(v.z, v.w));
}

template <typename VecT, typename ScalarT>
__device__ __forceinline__ VecT get_vec(const ScalarT* ptr) {
    return *reinterpret_cast<const VecT*>(ptr);
}

template<const int WS>
__device__ __forceinline__ unsigned long long packFloatIndex(float value, int index) {
    // Convert the 32-bit float 'value' into its raw bit representation (unsigned int).
    unsigned int value_bits = __float_as_uint(value);
    // Cast the 32-bit 'index' into an unsigned int (we assume index fits in 32 bits).
    unsigned int index_bits = static_cast<unsigned int>(index);
    // Shift the float's bits left by 'WS' so that they occupy the high bits of 64-bit,
    // then OR with the index_bits in the low bits to form a single 64-bit value.
    return (static_cast<unsigned long long>(value_bits) << WS) | index_bits;
}

template<const int WS = 32>
__device__ __forceinline__ void unpackFloatIndex(unsigned long long packed, float &outValue, int &outIndex) {
    // Extract the float bits by shifting 'packed' right by 'WS'.
    // This reverses the left shift we did in packFloatIndex.
    unsigned int value_bits = static_cast<unsigned int>(packed >> WS);
    // Extract the index bits by masking the lower 32 bits of 'packed'.
    unsigned int index_bits = static_cast<unsigned int>(packed & 0xFFFFFFFFu);
    // Reinterpret the 32-bit pattern 'value_bits' back into a float.
    outValue = __uint_as_float(value_bits);
    // Cast the unsigned index_bits back to a signed int.
    outIndex = static_cast<int>(index_bits);
}
