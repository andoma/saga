#include <memory>
#include <algorithm>

#include "common.h"

using namespace std;

namespace saga {

class BatchNorm : public Layer {

public:
  BatchNorm(double epsilon,
            const Layer &prev,
            Network &n,
            const char *scale,
            const char *bias,
            const char *mean,
            const char *var)
    : epsilon_(epsilon)
    , input_(prev.output())
    , output_(std::make_unique<Tensor>(*input_))
  {
    prev.output()->allocate();

    const Size s(1, input_->c, 1, 1);
    auto dt = input_->type();

    scale_ = n.findTensor(scale, s, dt, 1.0f, 0.0f);
    bias_  = n.findTensor(bias,  s, dt, 0.0f, 0.0f);
    mean_  = n.findTensor(mean,  s, dt, 0.0f, 0.0f);
    var_   = n.findTensor(var,   s, dt, 1.0f, 0.0f);
  }

  Tensor *output() const override {
    return output_.get();
  }

  string name() const override {
    stringstream ss;

    ss <<  "Batchnorm " << input_->name()
       << " scale " << scale_->name()
       << " => " << output_->name();
    return ss.str();
  }

  void forward(const Network &n) override {

    float alpha = 1.0f, beta = 0.0f;

    chkCUDNN(cudnnBatchNormalizationForwardInference(n.cudnn_,
                                                     CUDNN_BATCHNORM_SPATIAL,
                                                     &alpha, &beta,
                                                     input_->desc(),
                                                     input_->deviceMem(),
                                                     output_->desc(),
                                                     output_->deviceMem(),
                                                     scale_->desc(),
                                                     scale_->deviceMem(),
                                                     bias_->deviceMem(),
                                                     mean_->deviceMem(),
                                                     var_->deviceMem(),
                                                     epsilon_));
  }

protected:

  const double epsilon_;
  const Tensor *input_;
  unique_ptr<Tensor> output_;
  shared_ptr<Tensor> scale_;
  shared_ptr<Tensor> bias_;
  shared_ptr<Tensor> mean_;
  shared_ptr<Tensor> var_;


};


class BatchNormBackProp : public BatchNorm {

public:
  BatchNormBackProp(double epsilon,
                    const Layer &prev,
                    Network &n,
                    float expavgf,
                    const char *scale,
                    const char *bias,
                    const char *mean,
                    const char *var)
    : BatchNorm(epsilon, prev, n, scale, bias, mean, var)
    , input_grad_(prev.gradient())
    , output_grad_(make_unique<Tensor>(*output_))
    , saved_mean_(*scale_.get())
    , saved_ivar_(*scale_.get())
    , scale_grad_(*scale_.get())
    , bias_grad_(*scale_.get())
    , scale_optimizer_(n.makeOptimizer(*scale_))
    , bias_optimizer_(n.makeOptimizer(*bias_))
    , expavgf_(expavgf)
  {
    if(prev.gradient())
      prev.gradient()->allocate();

    saved_mean_.allocate();
    saved_ivar_.allocate();
    scale_grad_.allocate();
    bias_grad_.allocate();
  }


  void forward(const Network &n) override {

    float alpha = 1.0f, beta = 0.0f;
#if 0
    scale_->dump("SCALE");
    bias_->dump("BIAS");
    mean_->dump("MEAN");
    var_->dump("VAR");
#endif
    chkCUDNN(cudnnBatchNormalizationForwardTraining(n.cudnn_,
                                                    CUDNN_BATCHNORM_SPATIAL,
                                                    &alpha, &beta,
                                                    input_->desc(),
                                                    input_->deviceMem(),
                                                    output_->desc(),
                                                    output_->deviceMem(),
                                                    scale_->desc(),
                                                    scale_->deviceMem(),
                                                    bias_->deviceMem(),
                                                    expavgf_,
                                                    mean_->deviceMem(),
                                                    var_->deviceMem(),
                                                    epsilon_,
                                                    saved_mean_.deviceMem(),
                                                    saved_ivar_.deviceMem()
                                                    ));
  }


  void backprop(const Network &n) override {

    if(input_grad_ == NULL)
      return;

    float alpha = 1.0f, beta = 0.0f;

    chkCUDNN(cudnnBatchNormalizationBackward(n.cudnn_,
                                             CUDNN_BATCHNORM_SPATIAL,
                                             &alpha, &beta,
                                             &alpha, &beta,
                                             input_->desc(),
                                             input_->deviceMem(),
                                             output_grad_->desc(),
                                             output_grad_->deviceMem(),
                                             input_grad_->desc(),
                                             input_grad_->deviceMem(),
                                             scale_->desc(),
                                             scale_->deviceMem(),
                                             scale_grad_.deviceMem(),
                                             bias_grad_.deviceMem(),
                                             epsilon_,
                                             saved_mean_.deviceMem(),
                                             saved_ivar_.deviceMem()));


    scale_optimizer_->optimize(*scale_, scale_grad_, n);
    bias_optimizer_->optimize(*bias_, scale_grad_, n);
  }

  Tensor *gradient() const {
    return output_grad_.get();
  }

private:
  const Tensor *input_grad_;
  unique_ptr<Tensor> output_grad_;

  Tensor saved_mean_;
  Tensor saved_ivar_;

  Tensor scale_grad_;
  Tensor bias_grad_;

  std::unique_ptr<Optimizer> scale_optimizer_;
  std::unique_ptr<Optimizer> bias_optimizer_;

  float expavgf_;

};

shared_ptr<Layer> makeBatchNorm(double epsilon,
                                const Layer &prev,
                                Network &n,
                                float expavgf,
                                const char *scale,
                                const char *bias,
                                const char *mean,
                                const char *variance)
{
  if(n.backprop_) {
    return make_shared<BatchNormBackProp>(epsilon, prev, n, expavgf,
                                          scale, bias, mean, variance);
  } else {
    return make_shared<BatchNorm>(epsilon, prev, n,
                                  scale, bias, mean, variance);
  }
}


}
