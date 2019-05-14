#include <memory>
#include <algorithm>

#include "common.h"

namespace saga {


class FullyConnected : public Layer {

public:
  FullyConnected(int num_outputs, std::shared_ptr<Tensor> input,
                 const InitData &id)
    : num_inputs_(input->size().c * input->size().h * input->size().w)
    , num_outputs_(num_outputs)
    , input_(input)
    , weights_(input->dataType(), Size(num_inputs_, num_outputs, 1, 1))
    , bias_(input->dataType(), Size(1, num_outputs, 1, 1))
    , output_(Tensor::make(input->dataType(), Size(input->size().n,
                                                   num_outputs, 1, 1)))
  {
    weights_.loadOrRandomize(id, "weights", sqrt(1.0 / num_inputs_));
    bias_.loadOrRandomize(id, "bias", 0);
  }



  std::shared_ptr<Tensor> output() const override {
    return output_;
  }

  std::string name() const override {
    std::stringstream ss;
    ss << "FC " << input_->name() << " => " << output_->name();
    return ss.str();
  }

  // Replace this with cuTensor?

  void forward(const Network &n) override {
    float alpha = 1.0f, beta = 0.0f;

    chkCuda(cublasSgemm(n.cublas_, CUBLAS_OP_T, CUBLAS_OP_N,
                        num_outputs_, n.batch_size_, num_inputs_,
                        &alpha,
                        (const float *)weights_.deviceMem(), num_inputs_,
                        (const float *)input_->deviceMem(), num_inputs_,
                        &beta,
                        (float *)output_->deviceMem(), num_outputs_));

    chkCUDNN(cudnnAddTensor(n.cudnn_,
                            &alpha, bias_.desc(), bias_.deviceMem(),
                            &alpha, output_->desc(), output_->deviceMem()));
  }


protected:

  const int num_inputs_;
  const int num_outputs_;

  const std::shared_ptr<Tensor> input_;
  Tensor weights_;
  Tensor bias_;
  const std::shared_ptr<Tensor> output_;
};


class FullyConnectedBackProp : public FullyConnected {

public:
  FullyConnectedBackProp(int num_outputs, std::shared_ptr<Tensor> input,
                         const InitData &id, const Network &n)
    : FullyConnected(num_outputs, input, id)
    , input_grad_(Tensor::make(*input))
    , weights_grad_(weights_)
    , bias_grad_(bias_)
    , batch_of_one_(input->dataType(), Size(n.batch_size_, 1, 1, 1))
    , weights_optimizer_(n.makeOptimizer(weights_.size()))
    , bias_optimizer_(n.makeOptimizer(bias_.size()))
  {
    batch_of_one_.fill(1.0f);
  }



  std::shared_ptr<Tensor> backprop(const Network &n,
                                   const Tensor &dy) override {
    float alpha = 1.0f, beta = 0.0f;

    chkCuda(cublasSgemm(n.cublas_, CUBLAS_OP_N, CUBLAS_OP_T,
                        num_inputs_, num_outputs_, n.batch_size_,
                        &alpha,
                        (const float *)input_->deviceMem(), num_inputs_,
                        (const float *)dy.deviceMem(), num_outputs_,
                        &beta,
                        (float *)weights_grad_.deviceMem(), num_inputs_));

    chkCuda(cublasSgemv(n.cublas_, CUBLAS_OP_N, num_outputs_, n.batch_size_,
                        &alpha,
                        (const float *)dy.deviceMem(), num_outputs_,
                        (const float *)batch_of_one_.deviceMem(), 1,
                        &beta,
                        (float *)bias_grad_.deviceMem(), 1));

    chkCuda(cublasSgemm(n.cublas_, CUBLAS_OP_N, CUBLAS_OP_N,
                        num_inputs_, n.batch_size_, num_outputs_,
                        &alpha,
                        (const float *)weights_.deviceMem(), num_inputs_,
                        (const float *)dy.deviceMem(), num_outputs_,
                        &beta,
                        (float *)input_grad_->deviceMem(), num_inputs_));

    weights_optimizer_->optimize(weights_, weights_grad_, n);
    bias_optimizer_->optimize(bias_, bias_grad_, n);


    return input_grad_;
  }


protected:
  const std::shared_ptr<Tensor> input_grad_;

  const Tensor weights_grad_;
  const Tensor bias_grad_;
  Tensor batch_of_one_;

  std::unique_ptr<Optimizer> weights_optimizer_;
  std::unique_ptr<Optimizer> bias_optimizer_;

};


std::shared_ptr<Layer> makeFullyConnected(int num_outputs,
                                          std::shared_ptr<Tensor> input,
                                          const InitData &id,
                                          const Network &n)
{
  if(n.backprop_)
    return std::make_shared<FullyConnectedBackProp>(num_outputs, input, id, n);
  else
    return std::make_shared<FullyConnected>(num_outputs, input, id);
}


}
