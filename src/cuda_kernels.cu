#include "cuda_kernels.h"

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


namespace saga {


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


};
