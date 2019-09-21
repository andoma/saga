#include <assert.h>
#include <memory>
#include <algorithm>

#include "common.h"

using namespace std;

namespace saga {

class Softmax : public Layer {

public:
  Softmax(const Layer &prev)
    : input_(prev.output())
    , output_(make_unique<Tensor>(*input_))
  {
    prev.output()->allocate();
  }

  Tensor *output() const override {
    return output_.get();
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
  const Tensor *input_;
  unique_ptr<Tensor> output_;
};


class SoftmaxBackProp : public Softmax {
public:
  SoftmaxBackProp(const Layer &prev)
    : Softmax(prev)
    , input_grad_(prev.gradient())
    , output_grad_(make_unique<Tensor>(*output_))
  {
    prev.gradient()->allocate();
  }

  void backprop(const Network &n) override {

    float alpha = -1.0f, beta = 0.0f;

    chkCUDNN(cudnnSoftmaxBackward(n.cudnn_, CUDNN_SOFTMAX_ACCURATE,
                                  CUDNN_SOFTMAX_MODE_CHANNEL,
                                  &alpha,
                                  output_->desc(), output_->deviceMem(),
                                  output_grad_->desc(), output_grad_->deviceMem(),
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



std::shared_ptr<Layer> makeSoftmax(const Layer &prev,
                                   const Network &n)
{
  if(n.backprop_)
    return std::make_shared<SoftmaxBackProp>(prev);
  else
    return std::make_shared<Softmax>(prev);
}

}

