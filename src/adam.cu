#include <assert.h>
#include <memory>
#include <algorithm>

#include "common.h"

#define ADAM_EPSILON 10e-8
#define ADAM_B1      0.9
#define ADAM_B2      0.999


__global__
static void
adam(int n, float alpha, float b1t, float b2t, float *w,
     const float *dw, float *t)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i >= n)
    return;

  const float b1 = ADAM_B1;
  const float b2 = ADAM_B2;
  const float e = ADAM_EPSILON;

  t[i * 2 + 0] = b1 * t[i * 2 + 0] + (1.0f - b1) * dw[i];
  t[i * 2 + 1] = b2 * t[i * 2 + 1] + (1.0f - b2) * dw[i] * dw[i];
  w[i] += alpha * (t[i * 2 + 0] * b1t) / sqrtf(t[i * 2 + 1] * b2t + e);
}

namespace saga {


class Adam : public Optimizer {

public:
  Adam(const Size &s, const Network &net)
  {
    const size_t bytes = s.elements() * 2 * sizeof(float);
    chkCuda(cudaMalloc(&temp_, bytes));
  }

  ~Adam()
  {
    chkCuda(cudaFree(temp_));
  }

  void optimize(Tensor &x, const Tensor &grad, const Network &n,
                unsigned int iter) override {

    const float b1t = 1.0 / (1.0 - pow(ADAM_B1, 1 + iter));
    const float b2t = 1.0 / (1.0 - pow(ADAM_B2, 1 + iter));
    float alpha = -0.001 * powf(1.0 + 0.0001 * iter, -0.75);

    const size_t elements = grad.elements();

    adam<<<(elements+255)/256, 256>>>(elements,
                                      alpha,
                                      b1t,
                                      b2t,
                                      (float *)x.deviceMem(),
                                      (const float *)grad.deviceMem(),
                                      (float *)temp_);
  }

  float *temp_;
};


std::unique_ptr<Optimizer> makeAdamOptimizer(const Size &s,
                                             const Network &net)
{
  return std::make_unique<Adam>(s, net);
}


}
