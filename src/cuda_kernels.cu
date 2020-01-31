#include <stdint.h>
#include "cuda_kernels.h"

namespace saga {

//------------------------------------------------------------------------
// Category Classifier
//------------------------------------------------------------------------

template< typename T, typename L > __global__ static void
catclassifier_fwd(int n, const T *x, L *y, unsigned int channels)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i >= n)
    return;

  x += channels * i;

  L label = 0;
  T best = x[0];

  for(unsigned int j = 1; j < channels; j++) {
    if(x[j] > best) {
      best = x[j];
      label = j;
    }
  }
  y[i] = label;
}


template< typename T, typename L > __global__ static void
catclassifier_bwd(int n, const T *x, T *dx, const L *y, const L *dy,
                  float *loss, unsigned int channels, float scale)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i >= n)
    return;

  x  += channels * i;
  dx += channels * i;

  // Softmax
  const double max = x[y[i]];
  double sum = 0;
  for(unsigned int j = 0; j < channels; j++) {
    sum += exp((double)x[j] - max);
  }
  const double offset = max + log(sum);

  L label = dy[i];
  for(unsigned int j = 0; j < channels; j++) {

    const double p = exp((double)x[j] - offset);

    if(label == j) {
      dx[j] = (p - 1.0f) * scale;
      loss[i] = -log(p);
    } else {
      dx[j] = p * scale;
    }
  }
}




void
catclassifier_fwd_float_i32(int n, const float *x, int32_t *y, unsigned int c)
{
  catclassifier_fwd<<<(n+255)/256, 256>>>(n, x, y, c);
}

void
catclassifier_bwd_float_i32(int n, const float *x, float *dx,
                            const int32_t *y, const int32_t *dy,
                            float *loss, unsigned int c, float scale)
{
  catclassifier_bwd<<<(n+255)/256, 256>>>(n, x, dx, y, dy, loss, c, scale);
}


void
catclassifier_fwd_half_i32(int n, const __half *x, int32_t *y, unsigned int c)
{
  catclassifier_fwd<<<(n+255)/256, 256>>>(n, x, y, c);
}

void
catclassifier_bwd_half_i32(int n, const __half *x, __half *dx,
                           const int32_t *y, const int32_t *dy,
                           float *loss, unsigned int c, float scale)
{
  catclassifier_bwd<<<(n+255)/256, 256>>>(n, x, dx, y, dy, loss, c, scale);
}


//------------------------------------------------------------------------
// Datatype conversion
//------------------------------------------------------------------------

template< typename S, typename D > __global__ static void
convert(int n, const S *src, D *dst, float scale)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i >= n)
    return;

  dst[i] = src[i] * scale;
}


void
convert_u8_float(const void *src, void *dst, int elements, float scale)
{
  convert<<<(elements+255)/256, 256>>>(elements, (const uint8_t *)src, (float *)dst, scale);
}

void
convert_u8_half(const void *src, void *dst, int elements, float scale)
{
  convert<<<(elements+255)/256, 256>>>(elements, (const uint8_t *)src,
                                       (__half *)dst, scale);
}

void
convert_float_half(const void *src, void *dst, int elements, float scale)
{
  convert<<<(elements+255)/256, 256>>>(elements, (const float *)src,
                                       (__half *)dst, scale);
}

//------------------------------------------------------------------------
// Adam weight update
//------------------------------------------------------------------------

#define ADAM_EPSILON 1e-8
#define ADAM_B1      0.9
#define ADAM_B2      0.999

__global__ static void
adam_kernel(int n, float *weights, const float *dweights, float *t,
            float b1t, float b2t, float lr)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i >= n)
    return;

  t += i * 2; // mt and vt are stored next to each other in RAM

  const float b1 = ADAM_B1;
  const float b2 = ADAM_B2;
  const float e = ADAM_EPSILON;
  const float dw = dweights[i];

  const float m = t[0] = b1 * t[0] + (1.0f - b1) * dw;
  const float v = t[1] = b2 * t[1] + (1.0f - b2) * dw * dw;
  const float m_hat = m * b1t;
  const float v_hat = v * b2t;

  weights[i] -= lr * m_hat / (sqrtf(v_hat) + e);
}


__global__ static void
adam_kernel_mp(int n, float alpha, __half *weights, const __half *dweights,
               float *t, float b1t, float b2t, float lr, int *range)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i >= n)
    return;

  const uint16_t u16 = ((const uint16_t *)dweights)[i];
  if((u16 & 0x7800) == 0x7800) {
    *range = 1;
    if((u16 & 0x7c00) == 0x7c00) {
      // NaN or inf
      return;
    }
  }

  t += i * 3;

  const float dw = (float)dweights[i] * alpha;

  const float b1 = ADAM_B1;
  const float b2 = ADAM_B2;
  const float e = ADAM_EPSILON;

  const float m = t[0] = b1 * t[0] + (1.0f - b1) * dw;
  const float v = t[1] = b2 * t[1] + (1.0f - b2) * dw * dw;
  const float m_hat = m * b1t;
  const float v_hat = v * b2t;

  const float w = t[2] - lr * m_hat / (sqrtf(v_hat) + e);
  t[2] = w;
  weights[i] = w;
}

void
adam_float(int n, float *weights, const float *dweights, float *t,
           float b1t, float b2t, float lr)
{
  adam_kernel<<<(n+255)/256, 256>>>(n, weights, dweights, t, b1t, b2t, lr);
}

void
adam_mixed(int n, float alpha, __half *weights, const __half *dweights,
           float *t, float b1t, float b2t, float lr, int *range)
{
  adam_kernel_mp<<<(n+255)/256, 256>>>(n, alpha, weights, dweights, t, b1t, b2t,
                                       lr, range);
}

};
