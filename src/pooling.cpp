#include <assert.h>
#include <memory>
#include <algorithm>

#include "common.h"

namespace saga {


class Pooling : public Layer {

public:
  Pooling(PoolingMode mode, int size, int stride,
          const TensorDescriptor &input)
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
                                               input.desc(),
                                               &on, &oc, &oh, &ow));

    output_ = std::make_unique<Tensor>(TensorDescriptor(input.dataType(),
                                                        Size(on, oc, oh, ow)));
  }

  const Tensor *output() const override {
    return output_.get();
  }

  std::string name() const override {
    std::stringstream ss;
    ss << "Pooling " << input_.name() << " => " << output_->name();
    return ss.str();
  }


  const Tensor *forward(const Network &n,
                        const Tensor &input,
                        bool inference) override {
    float alpha = 1.0f, beta = 0.0f;

    chkCUDNN(cudnnPoolingForward(n.cudnn_, desc_,
                                 &alpha,
                                 input.desc(), input.deviceMem(),
                                 &beta,
                                 output_->desc(), output_->deviceMem()));
    return output_.get();
  }


protected:
  const TensorDescriptor input_;
  std::unique_ptr<Tensor> output_;
  cudnnPoolingDescriptor_t desc_;
};


class PoolingBackProp : public Pooling {
public:
  PoolingBackProp(PoolingMode mode, int size, int stride,
                  const TensorDescriptor input)
    : Pooling(mode, size, stride, input)
    , input_grad_(input)
  {}


  const Tensor *backprop(const Network &n,
                         const Tensor &input,
                         const Tensor &dy,
                         unsigned int iteration) override {
    float alpha = 1.0f, beta = 0.0f;

    chkCUDNN(cudnnPoolingBackward(n.cudnn_, desc_,
                                  &alpha,
                                  output_->desc(), output_->deviceMem(),
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



std::shared_ptr<Layer> makePooling(PoolingMode mode, int size, int stride,
                                   const TensorDescriptor &input,
                                   const Network &n)
{
  if(n.backprop_)
    return std::make_shared<PoolingBackProp>(mode, size, stride, input);
  else
    return std::make_shared<Pooling>(mode, size, stride, input);
}


}

