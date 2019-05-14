#include <assert.h>
#include <memory>
#include <algorithm>

#include "common.h"

namespace saga {


class Softmax : public Layer {

public:
  Softmax(std::shared_ptr<Tensor> input)
    : input_(input)
    , output_(Tensor::make(*input))
  {}

  std::shared_ptr<Tensor> output() const override {
    return output_;
  }

  std::string name() const override {
    std::stringstream ss;
    ss << "Softmax " << input_->name() << " => " << output_->name();
    return ss.str();
  }


  void forward(const Network &n) override {
    float alpha = 1.0f, beta = 0.0f;

    chkCUDNN(cudnnSoftmaxForward(n.cudnn_,
                                 CUDNN_SOFTMAX_ACCURATE,
                                 CUDNN_SOFTMAX_MODE_CHANNEL,
                                 &alpha,
                                 input_->desc(), input_->deviceMem(),
                                 &beta,
                                 output_->desc(), output_->deviceMem()));
  }

protected:
  const std::shared_ptr<Tensor> input_;
  const std::shared_ptr<Tensor> output_;
};


class SoftmaxBackProp : public Softmax {
public:
  SoftmaxBackProp(std::shared_ptr<Tensor> input)
    : Softmax(input)
    , input_grad_(Tensor::make(*input))
  {}


  std::shared_ptr<Tensor> backprop(const Network &n,
                                   const Tensor &dy) override {
    float alpha = -1.0f, beta = 0.0f;

    chkCUDNN(cudnnSoftmaxBackward(n.cudnn_, CUDNN_SOFTMAX_ACCURATE,
                                  CUDNN_SOFTMAX_MODE_CHANNEL,
                                  &alpha,
                                  output_->desc(), output_->deviceMem(),
                                  dy.desc(), dy.deviceMem(),
                                  &beta,
                                  input_grad_->desc(),
                                  input_grad_->deviceMem()));


    return input_grad_;
  }

protected:
  const std::shared_ptr<Tensor> input_grad_;
};



std::shared_ptr<Layer> makeSoftmax(std::shared_ptr<Tensor> input,
                                   const Network &n)
{
  if(n.backprop_)
    return std::make_shared<SoftmaxBackProp>(input);
  else
    return std::make_shared<Softmax>(input);
}

}

