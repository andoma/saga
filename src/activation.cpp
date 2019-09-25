#include <memory>
#include <algorithm>

#include "common.h"

using namespace std;

namespace saga {

class Activation : public Layer {

public:
  Activation(ActivationMode mode, float a,
             const Layer &prev)
    : input_(prev.output())
    , output_(make_unique<Tensor>(*input_))
  {
    prev.output()->allocate();

    cudnnActivationMode_t cudnn_mode;
    switch(mode) {
    case ActivationMode::RELU:
      cudnn_mode = CUDNN_ACTIVATION_RELU;
      break;
    case ActivationMode::ELU:
      cudnn_mode = CUDNN_ACTIVATION_ELU;
      break;
    default:
      abort();
    }

    chkCUDNN(cudnnCreateActivationDescriptor(&desc_));
    chkCUDNN(cudnnSetActivationDescriptor(desc_, cudnn_mode,
                                          CUDNN_PROPAGATE_NAN, a));
  }

  Tensor *output() const override {
    return output_.get();
  }

  std::string name() const override {
    std::stringstream ss;
    ss << "Activation " << input_->name() << " => " << output_->name();
    return ss.str();
  }


  void forward(const Network &n) override {

    float alpha = 1.0f, beta = 0.0f;

    chkCUDNN(cudnnActivationForward(n.cudnn_, desc_,
                                    &alpha,
                                    input_->desc(), input_->deviceMem(),
                                    &beta,
                                    output_->desc(), output_->deviceMem()));
  }


protected:
  const Tensor *input_;
  unique_ptr<Tensor> output_;

  cudnnActivationDescriptor_t desc_;
};


class ActivationBackProp : public Activation {
public:
  ActivationBackProp(ActivationMode mode, float a, const Layer &prev)
    : Activation(mode, a, prev)
    , input_grad_(prev.gradient())
    , output_grad_(make_unique<Tensor>(*output_))
  {
    prev.gradient()->allocate();
  }


  void backprop(const Network &n) override {

    float alpha = 1.0f, beta = 0.0f;

    chkCUDNN(cudnnActivationBackward(n.cudnn_, desc_,
                                     &alpha,
                                     output_->desc(), output_->deviceMem(),
                                     output_grad_->desc(), output_grad_->deviceMem(),
                                     input_->desc(), input_->deviceMem(),
                                     &beta,
                                     input_grad_->desc(),
                                     input_grad_->deviceMem()));
  }

  Tensor *gradient() const {
    return output_grad_.get();
  }

protected:
  const Tensor *input_grad_;
  unique_ptr<Tensor> output_grad_;
};



std::shared_ptr<Layer> makeActivation(ActivationMode mode, float a,
                                      const Layer &prev,
                                      const Network &n)
{
  if(n.backprop_)
    return std::make_shared<ActivationBackProp>(mode, a, prev);
  else
    return std::make_shared<Activation>(mode, a, prev);
}


}

