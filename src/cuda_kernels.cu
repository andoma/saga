#include "cuda_kernels.h"

namespace saga {

//------------------------------------------------------------------------
// Category Classifier
//------------------------------------------------------------------------

template< typename T, typename L > __global__ static void
catclassifier_pred(int n, const T *input, L *output, unsigned int channels)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i >= n)
    return;

  input  += channels * i;

  L label = 0;
  T best = input[0];

  for(unsigned int j = 1; j < channels; j++) {
    if(input[j] > best) {
      best = input[j];
      label = j;
    }
  }
  output[i] = label;
}


template< typename T, typename L > __global__ static void
catclassifier_backprop(int n, const T *prob, T *grad, const L *labels, float *loss,
                       unsigned int channels, float scale)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i >= n)
    return;

  prob += channels * i;
  grad += channels * i;

  L label = labels[i];
  for(unsigned int j = 0; j < channels; j++) {
    grad[j] = static_cast<T>((static_cast<float>(prob[j]) - (label == j ? 1.0f : 0.0f)) * scale);
  }
  if(loss)
    loss[i] = -log(static_cast<float>(prob[label]));
}




void
catclassifier_pred_float_i32(int n, const float *p, int32_t *y, unsigned int c)
{
  catclassifier_pred<<<(n+255)/256, 256>>>(n, p, y, c);
}

void
catclassifier_backprop_float_i32(int n, const float *p, float *dx,
                                 const int32_t *dy, float *loss,
                                 unsigned int c, float scale)
{
  catclassifier_backprop<<<(n+255)/256, 256>>>(n, p, dx, dy, loss, c, scale);
}



//------------------------------------------------------------------------
// Adam weight update
//------------------------------------------------------------------------

#define ADAM_EPSILON 1e-8
#define ADAM_B1      0.9
#define ADAM_B2      0.999

__global__ static void
adam_kernel(int n, float alpha, float *weights, const float *dweights, float *t,
            float b1t, float b2t)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i >= n)
    return;

  weights[i] -= alpha * dweights[i];
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

  weights[i] -= alpha * m_hat / (sqrtf(v_hat) + e);
}

void
adam_float(int n, float alpha, float *weights, const float *dweights, float *t,
           float b1t, float b2t)
{
  adam_kernel<<<(n+255)/256, 256>>>(n, alpha, weights, dweights, t, b1t, b2t);
}


};
