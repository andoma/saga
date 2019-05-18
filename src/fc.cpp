#include <memory>
#include <algorithm>

#include "common.h"

namespace saga {


class FullyConnected : public Layer {

public:
  FullyConnected(int num_outputs,
                 const TensorDescriptor &input,
                 const InitData &id)
    : num_inputs_(input.c * input.h * input.w)
    , num_outputs_(num_outputs)
    , input_(input)
    , weights_(TensorDescriptor(input.dataType(),
                                input.format(),
                                Size(num_inputs_, num_outputs, 1, 1)))
    , bias_(TensorDescriptor(input.dataType(),
                             input.format(),
                             Size(1, num_outputs, 1, 1)))
    , output_(TensorDescriptor(input.dataType(),
                               input.format(),
                               Size(input.n, num_outputs, 1, 1)))
  {
    weights_.loadOrRandomize(id, "weights", sqrt(1.0 / num_inputs_));
    bias_.loadOrRandomize(id, "bias", 0);
  }



  const Tensor *output() const override {
    return &output_;
  }

  std::string name() const override {
    std::stringstream ss;
    ss << "FC " << input_.name() << " => " << output_.name();
    return ss.str();
  }

  // Replace this with cuTensor?

  const Tensor *forward(const Network &n,
                        const Tensor &input,
                        bool inference) override {
    float alpha = 1.0f, beta = 0.0f;

    chkCuda(cublasSgemm(n.cublas_, CUBLAS_OP_T, CUBLAS_OP_N,
                        num_outputs_, n.batch_size_, num_inputs_,
                        &alpha,
                        (const float *)weights_.deviceMem(), num_inputs_,
                        (const float *)input.deviceMem(), num_inputs_,
                        &beta,
                        (float *)output_.deviceMem(), num_outputs_));

    chkCUDNN(cudnnAddTensor(n.cudnn_,
                            &alpha, bias_.desc(), bias_.deviceMem(),
                            &alpha, output_.desc(), output_.deviceMem()));
    return &output_;
  }


protected:

  const int num_inputs_;
  const int num_outputs_;

  const TensorDescriptor input_;
  Tensor weights_;
  Tensor bias_;
  Tensor output_;
};


class FullyConnectedBackProp : public FullyConnected {

public:
  FullyConnectedBackProp(int num_outputs,
                         const TensorDescriptor &input,
                         const InitData &id,
                         const Network &n)
    : FullyConnected(num_outputs, input, id)
    , input_grad_(input)
    , weights_grad_(TensorDescriptor(weights_))
    , bias_grad_(TensorDescriptor(bias_))
    , batch_of_one_(TensorDescriptor(input.dataType(),
                                     input.format(),
                                     Size(n.batch_size_, 1, 1, 1)))
    , weights_optimizer_(n.makeOptimizer(weights_))
    , bias_optimizer_(n.makeOptimizer(bias_))
  {
    batch_of_one_.fill(1.0f);
  }



  const Tensor *backprop(const Network &n,
                         const Tensor &input,
                         const Tensor &dy,
                         unsigned int iteration) override {
    float alpha = 1.0f, beta = 0.0f;

    chkCuda(cublasSgemm(n.cublas_, CUBLAS_OP_N, CUBLAS_OP_T,
                        num_inputs_, num_outputs_, n.batch_size_,
                        &alpha,
                        (const float *)input.deviceMem(), num_inputs_,
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
                        (float *)input_grad_.deviceMem(), num_inputs_));

    weights_optimizer_->optimize(weights_, weights_grad_, n, iteration);
    bias_optimizer_->optimize(bias_, bias_grad_, n, iteration);

    return &input_grad_;
  }


protected:
  const Tensor input_grad_;
  const Tensor weights_grad_;
  const Tensor bias_grad_;

  Tensor batch_of_one_;

  std::unique_ptr<Optimizer> weights_optimizer_;
  std::unique_ptr<Optimizer> bias_optimizer_;

};


std::shared_ptr<Layer> makeFullyConnected(int num_outputs,
                                          const TensorDescriptor &input,
                                          const InitData &id,
                                          const Network &n)
{
  if(n.backprop_)
    return std::make_shared<FullyConnectedBackProp>(num_outputs, input, id, n);
  else
    return std::make_shared<FullyConnected>(num_outputs, input, id);
}


}
