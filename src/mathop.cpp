#include <memory>
#include <algorithm>

#include "common.h"

using namespace std;

namespace saga {


class MathOp : public Layer {

public:
  MathOp(const Layer &prev,
         cudnnOpTensorOp_t op,
         shared_ptr<Tensor> b,
         float alpha1,
         float alpha2,
         Network &net)
    : input_(prev.output())
    , b_(b)
    , alpha1_(alpha1)
    , alpha2_(alpha2)
    , output_(make_unique<Tensor>(*prev.output()))
  {
    prev.output()->allocate();
    chkCUDNN(cudnnCreateOpTensorDescriptor(&desc_));
    chkCUDNN(cudnnSetOpTensorDescriptor(desc_,
                                        op, input_->cudnnType(),
                                        CUDNN_PROPAGATE_NAN));
  }

  Tensor *output() const override {
    return output_.get();
  }

  string name() const override {
    stringstream ss;
    ss << "Math " << input_->name() << " => " << output_->name();
    return ss.str();
  }

  void forward(const Network &n) override {
    float beta = 0.0f;
    chkCUDNN(cudnnOpTensor(n.cudnn_, desc_,
                           &alpha1_,
                           input_->desc(), input_->deviceMem(),
                           &alpha2_,
                           b_->desc(), b_->deviceMem(),
                           &beta,
                           output_->desc(),
                           output_->deviceMem()));
  }


protected:

  Tensor *input_;

  shared_ptr<Tensor> b_;
  float alpha1_;
  float alpha2_;

  unique_ptr<Tensor> output_;
  cudnnOpTensorDescriptor_t desc_;
};


std::shared_ptr<Layer> makeMathOp(const Layer &prev,
                                  cudnnOpTensorOp_t op,
                                  shared_ptr<Tensor> b,
                                  float alpha1,
                                  float alpha2,
                                  Network &net)
{
  return std::make_shared<MathOp>(prev, op, b, alpha1, alpha2, net);
}

}
