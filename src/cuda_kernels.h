// -*-c++-*-

#include <cuda_fp16.h>

#pragma once

namespace saga {

void catclassifier_fwd_float_i32(int n, const float *p,
                                 int32_t *y, unsigned int c);

void catclassifier_fwd_half_i32(int n, const __half *p,
                                int32_t *y, unsigned int c);

void catclassifier_bwd_float_i32(int n, const float *x, float *dx,
                                 const int32_t *y, const int32_t *dy,
                                 float *loss, unsigned int c, float scale);

void catclassifier_bwd_half_i32(int n, const __half *x, __half *dx,
                                const int32_t *y, const int32_t *dy,
                                float *loss, unsigned int c, float scale);

void convert_u8_float(const void *src, void *dst, int elements, float scale);

void convert_u8_half(const void *src, void *dst, int elements, float scale);

void convert_float_half(const void *src, void *dst, int elements, float scale);

void adam_float(int n, float alpha, float *weights,
                const float *dweights, float *t,
                float b1t, float b2t);

void adam_mixed(int n, float alpha, __half *weights,
                const __half *dweights,
                float *t, float b1t, float b2t);

}
