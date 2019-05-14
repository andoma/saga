#include <assert.h>
#include <memory>
#include <algorithm>

#include "common.h"

namespace saga {


class Pooling : public Layer {

public:
  Pooling(PoolingMode mode, int size, int stride,
          std::shared_ptr<Tensor> input)
    : input_(input)
  {
    assert(mode == PoolingMode::MAX);

    chkCUDNN(cudnnCreatePoolingDescriptor(&desc_));
    chkCUDNN(cudnnSetPooling2dDescriptor(desc_,
                                         CUDNN_POOLING_MAX,
                                         CUDNN_PROPAGATE_NAN,
                                         size, size,
                                         0, 0,
                                         stride, stride));

    int on, oc, oh, ow;
    chkCUDNN(cudnnGetPooling2dForwardOutputDim(desc_,
                                               input_->desc(),
                                               &on, &oc, &oh, &ow));

    output_ = Tensor::make(input->dataType(), Size(on, oc, oh, ow));
  }

  std::shared_ptr<Tensor> output() const override {
    return output_;
  }

  std::string name() const override {
    std::stringstream ss;
    ss << "Pooling " << input_->name() << " => " << output_->name();
    return ss.str();
  }


  void forward(const Network &n) override {
    float alpha = 1.0f, beta = 0.0f;

    chkCUDNN(cudnnPoolingForward(n.cudnn_, desc_,
                                 &alpha,
                                 input_->desc(), input_->deviceMem(),
                                 &beta,
                                 output_->desc(), output_->deviceMem()));
  }


protected:
  const std::shared_ptr<Tensor> input_;
  std::shared_ptr<Tensor> output_;
  cudnnPoolingDescriptor_t desc_;
};


class PoolingBackProp : public Pooling {
public:
  PoolingBackProp(PoolingMode mode, int size, int stride,
                  std::shared_ptr<Tensor> input)
    : Pooling(mode, size, stride, input)
    , input_grad_(Tensor::make(*input))
  {}


  std::shared_ptr<Tensor> backprop(const Network &n,
                                   std::shared_ptr<Tensor> dy) override {
    float alpha = 1.0f, beta = 0.0f;

    chkCUDNN(cudnnPoolingBackward(n.cudnn_, desc_,
                                  &alpha,
                                  output_->desc(), output_->deviceMem(),
                                  dy->desc(), dy->deviceMem(),
                                  input_->desc(), input_->deviceMem(),
                                  &beta,
                                  input_grad_->desc(),
                                  input_grad_->deviceMem()));
    return input_grad_;
  }

protected:
  const std::shared_ptr<Tensor> input_grad_;
};



std::shared_ptr<Layer> makePooling(PoolingMode mode, int size, int stride,
                                   std::shared_ptr<Tensor> input,
                                   const Network &n)
{
  if(n.backprop_)
    return std::make_shared<PoolingBackProp>(mode, size, stride, input);
  else
    return std::make_shared<Pooling>(mode, size, stride, input);
}


}

