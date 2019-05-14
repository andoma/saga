#include <assert.h>
#include <memory>
#include <algorithm>

#include "common.h"

namespace saga {


class GradientDescent : public Optimizer {

public:
  GradientDescent(const Size &s, const Network &net)
  {}

  void optimize(Tensor &x, const Tensor &grad, const Network &n) override {

    assert(x.size() == grad.size());

    const float learning_rate = 0.01 * powf(1.0 + 0.0001 * n.iter_, -0.75);
    const float alpha = -learning_rate;
    const float beta = 1.0f;

    chkCUDNN(cudnnAddTensor(n.cudnn_,
                            &alpha, grad.desc(), grad.deviceMem(),
                            &beta,   x.desc(),  x.deviceMem()));

  }

};


std::unique_ptr<Optimizer> makeGradientDescentOptimizer(const Size &s,
                                                        const Network &net)
{
  return std::make_unique<GradientDescent>(s, net);
}


}
