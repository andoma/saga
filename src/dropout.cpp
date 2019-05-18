#include <assert.h>
#include <memory>
#include <algorithm>

#include "common.h"

namespace saga {

class DropoutBackProp : public Layer {

public:
  DropoutBackProp(float prob,
                  const TensorDescriptor &input,
                  const Network &n)
    : input_(input)
    , input_grad_(input)
    , output_(input)
  {
    chkCUDNN(cudnnDropoutGetReserveSpaceSize(input.desc(), &reserve_size_));

    chkCuda(cudaMalloc(&reserve_, reserve_size_));

    chkCUDNN(cudnnDropoutGetStatesSize(n.cudnn_, &states_size_));
    chkCuda(cudaMalloc(&states_, states_size_));

    chkCUDNN(cudnnCreateDropoutDescriptor(&desc_));
    chkCUDNN(cudnnSetDropoutDescriptor(desc_, n.cudnn_, prob,
                                       states_, states_size_, 0)); //rand()));
  }

  const Tensor *forward(const Network &n,
                        const Tensor &input,
                        bool inference) override {
    if(inference) {
      return &input;
    }
    chkCUDNN(cudnnDropoutForward(n.cudnn_, desc_,
                                 input.desc(), input.deviceMem(),
                                 output_.desc(), output_.deviceMem(),
                                 reserve_, reserve_size_));
    return &output_;
  }

  const Tensor *backprop(const Network &n,
                         const Tensor &input,
                         const Tensor &dy,
                         unsigned int iteration) override {

    chkCUDNN(cudnnDropoutBackward(n.cudnn_, desc_,
                                  dy.desc(), dy.deviceMem(),
                                  input_grad_.desc(), input_grad_.deviceMem(),
                                  reserve_, reserve_size_));
    return &input_grad_;
  }

  const Tensor *output() const override {
    return &output_;
  }

  std::string name() const override {
    std::stringstream ss;
    ss << "Dropout " << input_.name() << " => " << output_.name();
    return ss.str();
  }

  const TensorDescriptor input_;
  Tensor input_grad_;

  Tensor output_;

  cudnnDropoutDescriptor_t desc_;
  size_t reserve_size_;
  void *reserve_;

  size_t states_size_;
  void *states_;
};



std::shared_ptr<Layer> makeDropout(float prob,
                                   const TensorDescriptor &input,
                                   const Network &n)
{
  if(n.backprop_)
    return std::make_shared<DropoutBackProp>(prob, input, n);
  else
    abort(); // Dropout in inference mode make no sense
}


}

