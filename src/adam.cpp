#include <assert.h>
#include <memory>
#include <algorithm>

#include "common.h"

namespace saga {


class Adam : public Optimizer {

public:
  Adam(const Size &s, const Network &net)
  {}

  void optimize(Tensor &x, const Tensor &grad, const Network &n) override {

    printf("I can't optimize much yet\n");
  }

};


std::unique_ptr<Optimizer> makeAdamOptimizer(const Size &s,
                                             const Network &net)
{
  return std::make_unique<Adam>(s, net);
}


}
