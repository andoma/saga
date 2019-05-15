#include <memory>
#include <algorithm>

#include "common.h"

namespace saga {


class Activation : public Layer {

public:
  Activation(ActivationMode mode, float a,
             const TensorDescriptor &input)
    : input_(input)
    , output_(input)
  {
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

  const Tensor *output() const override {
    return &output_;
  }

  std::string name() const override {
    std::stringstream ss;
    ss << "Activation " << input_.name() << " => " << output_.name();
    return ss.str();
  }

  const Tensor *forward(const Network &n,
                        const Tensor &input,
                        bool inference) override {
    float alpha = 1.0f, beta = 0.0f;

    chkCUDNN(cudnnActivationForward(n.cudnn_, desc_,
                                    &alpha,
                                    input.desc(), input.deviceMem(),
                                    &beta,
                                    output_.desc(), output_.deviceMem()));
    return &output_;
  }


protected:
  const TensorDescriptor input_;
  Tensor output_;

  cudnnActivationDescriptor_t desc_;
};


class ActivationBackProp : public Activation {
public:
  ActivationBackProp(ActivationMode mode, float a,
                     const TensorDescriptor &input)
    : Activation(mode, a, input)
    , input_grad_(input)
  {}


  const Tensor *backprop(const Network &n,
                         const Tensor &input,
                         const Tensor &dy) override {

    float alpha = 1.0f, beta = 0.0f;

    chkCUDNN(cudnnActivationBackward(n.cudnn_, desc_,
                                     &alpha,
                                     output_.desc(), output_.deviceMem(),
                                     dy.desc(), dy.deviceMem(),
                                     input.desc(), input.deviceMem(),
                                     &beta,
                                     input_grad_.desc(),
                                     input_grad_.deviceMem()));
    return &input_grad_;
  }

protected:
  Tensor input_grad_;
};



std::shared_ptr<Layer> makeActivation(ActivationMode mode, float a,
                                      const TensorDescriptor &input,
                                      const Network &n)
{
  if(n.backprop_)
    return std::make_shared<ActivationBackProp>(mode, a, input);
  else
    return std::make_shared<Activation>(mode, a, input);
}


}

