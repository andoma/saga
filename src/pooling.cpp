#include <assert.h>
#include <memory>
#include <algorithm>

#include "common.h"

namespace saga {


class Pooling : public Layer {

public:
  Pooling(PoolingMode mode, int size, int pad, int stride,
          const Layer &prev)
    : input_(prev.output())
  {
    cudnnPoolingMode_t cudnn_mode;
    switch(mode) {
    case PoolingMode::MAX:
      cudnn_mode = CUDNN_POOLING_MAX;
      break;
    case PoolingMode::AVERAGE:
      cudnn_mode = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
      break;
    default:
      abort();
    }

    chkCUDNN(cudnnCreatePoolingDescriptor(&desc_));
    chkCUDNN(cudnnSetPooling2dDescriptor(desc_,
                                         cudnn_mode,
                                         CUDNN_PROPAGATE_NAN,
                                         size, size,
                                         pad, pad,
                                         stride, stride));

    int on, oc, oh, ow;
    chkCUDNN(cudnnGetPooling2dForwardOutputDim(desc_,
                                               input_->desc(),
                                               &on, &oc, &oh, &ow));

    output_ = std::make_unique<Tensor>(TensorDescriptor(input_->dataType(),
                                                        input_->format(),
                                                        Size(on, oc, oh, ow)));
  }

  const Tensor *output() const override {
    return output_.get();
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
  const Tensor *input_;
  std::unique_ptr<Tensor> output_;
  cudnnPoolingDescriptor_t desc_;
};


class PoolingBackProp : public Pooling {
public:
  PoolingBackProp(PoolingMode mode, int size, int pad, int stride,
                  const Layer &prev)
    : Pooling(mode, size, pad, stride, prev)
    , input_grad_(prev.gradient())
    , output_grad_(*output_)
  {}


  void backprop(const Network &n) override {

    float alpha = 1.0f, beta = 0.0f;

    chkCUDNN(cudnnPoolingBackward(n.cudnn_, desc_,
                                  &alpha,
                                  output_->desc(), output_->deviceMem(),
                                  output_grad_.desc(), output_grad_.deviceMem(),
                                  input_->desc(), input_->deviceMem(),
                                  &beta,
                                  input_grad_->desc(),
                                  input_grad_->deviceMem()));
  }

  Tensor *gradient() const {
    return (Tensor *)&output_grad_;
  }

protected:
  const Tensor *input_grad_;
  const Tensor output_grad_;
};



std::shared_ptr<Layer> makePooling(PoolingMode mode, int size, int pad,
                                   int stride,
                                   const Layer &prev, const Network &n)
{
  if(n.backprop_)
    return std::make_shared<PoolingBackProp>(mode, size, pad, stride, prev);
  else
    return std::make_shared<Pooling>(mode, size, pad, stride, prev);
}


}

