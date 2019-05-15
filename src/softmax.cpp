#include <assert.h>
#include <memory>
#include <algorithm>

#include "common.h"

namespace saga {


class Softmax : public Layer {

public:
  Softmax(const TensorDescriptor &input)
    : input_(input)
    , output_(input)
  {}

  const Tensor *output() const override {
    return &output_;
  }

  std::string name() const override {
    std::stringstream ss;
    ss << "Softmax " << input_.name() << " => " << output_.name();
    return ss.str();
  }


  const Tensor *forward(const Network &n,
                        const Tensor &input,
                        bool inference) override {
    float alpha = 1.0f, beta = 0.0f;

    chkCUDNN(cudnnSoftmaxForward(n.cudnn_,
                                 CUDNN_SOFTMAX_ACCURATE,
                                 CUDNN_SOFTMAX_MODE_CHANNEL,
                                 &alpha,
                                 input.desc(), input.deviceMem(),
                                 &beta,
                                 output_.desc(), output_.deviceMem()));

    return &output_;
  }

protected:
  const TensorDescriptor input_;
  Tensor output_;
};


class SoftmaxBackProp : public Softmax {
public:
  SoftmaxBackProp(const TensorDescriptor &input)
    : Softmax(input)
    , input_grad_(input)
  {}


  const Tensor *backprop(const Network &n,
                         const Tensor &input,
                         const Tensor &dy) override {

    float alpha = -1.0f, beta = 0.0f;

    chkCUDNN(cudnnSoftmaxBackward(n.cudnn_, CUDNN_SOFTMAX_ACCURATE,
                                  CUDNN_SOFTMAX_MODE_CHANNEL,
                                  &alpha,
                                  output_.desc(), output_.deviceMem(),
                                  dy.desc(), dy.deviceMem(),
                                  &beta,
                                  input_grad_.desc(),
                                  input_grad_.deviceMem()));


    return &input_grad_;
  }

protected:
  Tensor input_grad_;
};



std::shared_ptr<Layer> makeSoftmax(const TensorDescriptor &input,
                                   const Network &n)
{
  if(n.backprop_)
    return std::make_shared<SoftmaxBackProp>(input);
  else
    return std::make_shared<Softmax>(input);
}

}

