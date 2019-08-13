#include <memory>
#include <algorithm>

#include "common.h"

namespace saga {

class BatchNorm : public Layer {

public:
  BatchNorm(double epsilon,
            const Layer &prev,
            const Network &n,
            std::shared_ptr<Tensor> scale,
            std::shared_ptr<Tensor> bias,
            std::shared_ptr<Tensor> mean,
            std::shared_ptr<Tensor> variance)
    : epsilon_(epsilon)
    , input_(prev.output())
    , output_(*input_)
  {
    scale_ = scale;
    bias_ = bias;
    mean_ = mean;
    variance_ = variance;
  }

  const Tensor *output() const override {
    return &output_;
  }

  std::string name() const override {
    std::stringstream ss;

    ss <<  "Batchnorm " << input_->name()
       << " scale " << scale_->name()
       << " => " << output_.name();
    return ss.str();
  }

  void forward(const Network &n) override {

    float alpha = 1.0f, beta = 0.0f;

    cudnnBatchNormalizationForwardInference(n.cudnn_, CUDNN_BATCHNORM_SPATIAL,
                                            &alpha, &beta,
                                            input_->desc(),
                                            input_->deviceMem(),
                                            output_.desc(),
                                            output_.deviceMem(),
                                            scale_->desc(),
                                            scale_->deviceMem(),
                                            bias_->deviceMem(),
                                            mean_->deviceMem(),
                                            variance_->deviceMem(),
                                            epsilon_);
  }

protected:

  const double epsilon_;
  const Tensor *input_;
  Tensor output_;
  std::shared_ptr<Tensor> scale_;
  std::shared_ptr<Tensor> bias_;
  std::shared_ptr<Tensor> mean_;
  std::shared_ptr<Tensor> variance_;

};





std::shared_ptr<Layer> makeBatchNorm(double epsilon,
                                     const Layer &prev,
                                     const Network &n,
                                     std::shared_ptr<Tensor> scale,
                                     std::shared_ptr<Tensor> bias,
                                     std::shared_ptr<Tensor> mean,
                                     std::shared_ptr<Tensor> variance)
{
  assert(n.backprop_ == 0);

  return std::make_shared<BatchNorm>(epsilon, prev, n,
                                     scale, bias, mean, variance);
}


}
