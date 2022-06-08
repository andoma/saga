// -*-c++-*-

#include <cuda_fp16.h>

#pragma once

namespace saga {

void catclassifier_fwd_float_i32(int n, const float *p, int32_t *y,
                                 unsigned int c, cudaStream_t stream);

void catclassifier_fwd_half_i32(int n, const __half *p, int32_t *y,
                                unsigned int c, cudaStream_t stream);

void catclassifier_bwd_float_i32(int n, const float *x, float *dx,
                                 const int32_t *y, const int32_t *dy,
                                 float *loss, unsigned int c, float scale,
                                 cudaStream_t stream);

void catclassifier_bwd_half_i32(int n, const __half *x, __half *dx,
                                const int32_t *y, const int32_t *dy,
                                float *loss, unsigned int c, float scale,
                                cudaStream_t stream);

void convert_u8_float(const void *src, void *dst, int elements, float scale,
                      cudaStream_t stream);

void convert_u8_half(const void *src, void *dst, int elements, float scale,
                     cudaStream_t stream);

void convert_i16_half(const void *src, void *dst, int elements, float scale,
                      cudaStream_t stream);

void convert_float_half(const void *src, void *dst, int elements, float scale,
                        cudaStream_t stream);

void adam_float(int n, float *weights, const float *dweights, float *t,
                float b1t, float b2t, float lr, cudaStream_t stream);

void adam_mixed(int n, float alpha, __half *weights, const __half *dweights,
                float *t, float b1t, float b2t, float lr, int *range,
                cudaStream_t stream);

// Compute min, max, sum and sum-of-squares and store in output[4]
void tensor_stats_float(int n, const float *src, float *output,
                        cudaStream_t stream);

void tensor_stats_half(int n, const __half *src, float *output,
                       cudaStream_t stream);

}  // namespace saga
