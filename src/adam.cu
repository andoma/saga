#include <assert.h>
#include <memory>
#include <algorithm>

#include "common.h"
#define ADAM_EPSILON 1e-8
#define ADAM_B1      0.9
#define ADAM_B2      0.999


__global__
static void
adam(int n, float alpha, float *weights, const float *dweights, float *t,
     float b1t, float b2t)
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

  weights[i] -= alpha * m_hat / (sqrtf(v_hat) + e);
}



namespace saga {


class Adam : public Optimizer {

  float *temp_;
  float lr_;
  int iter_;

public:
  Adam(const Size &s, const Network &net, float lr)
    : lr_(lr)
    , iter_(0)
  {
    const size_t bytes = s.elements() * 2 * sizeof(float);
    chkCuda(cudaMalloc(&temp_, bytes));
    chkCuda(cudaMemset(temp_, 0, bytes));
  }

  ~Adam()
  {
    chkCuda(cudaFree(temp_));
  }

  void optimize(Tensor &x, const Tensor &grad, const Network &n) override {

    const size_t elements = grad.elements();
    assert(elements == x.elements());

    const int i = ++iter_;

    const float b1t = 1.0 / (1.0 - pow(ADAM_B1, i));
    const float b2t = 1.0 / (1.0 - pow(ADAM_B2, i));
    adam<<<(elements+255)/256, 256>>>(elements, lr_,
                                      (float *)x.deviceMem(),
                                      (const float *)grad.deviceMem(),
                                      (float *)temp_, b1t, b2t);
  }
};


std::unique_ptr<Optimizer> makeAdamOptimizer(const Size &s,
                                             const Network &net,
                                             float lr)
{
  return std::make_unique<Adam>(s, net, lr);
}


}
