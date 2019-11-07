/*
 * Copyright (c) 2019, Andreas Smas
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <memory>
#include <algorithm>

#include "common.h"

using namespace std;

namespace saga {


class FullyConnected : public Layer {

public:
  FullyConnected(int num_outputs,
                 const Layer &prev,
                 Network &net,
                 const char *weights,
                 const char *bias)
    : input_(prev.output())
    , num_inputs_(input_->c * input_->h * input_->w)
    , num_outputs_(num_outputs)
    , output_(make_unique<Tensor>(Size(input_->n, num_outputs, 1, 1),
                                  input_->type()))
  {
    prev.output()->allocate();

    weights_ = net.findTensor(weights, Size(num_inputs_, num_outputs, 1, 1),
                              input_->type(),
                              0.0, sqrt(2.0 / num_inputs_));

    bias_ = net.findTensor(bias, Size(1, num_outputs, 1, 1), input_->type(),
                           0.0, 0.0f);
  }

  Tensor *output() const override {
    return output_.get();
  }

  string name() const override {
    stringstream ss;
    ss << "FC " << input_->name() << " => " << output_->name();
    return ss.str();
  }

  void forward(const Network &n) override {

    float alpha = 1.0f, beta = 0.0f;
    __half halpha = 1.0f, hbeta = 0.0f;

    switch(input_->type()) {
    case Tensor::Type::FLOAT:
      chkCuda(cublasSgemm(n.cublas_, CUBLAS_OP_T, CUBLAS_OP_N,
                          num_outputs_, input_->n, num_inputs_,
                          &alpha,
                          (const float *)weights_->deviceMem(), num_inputs_,
                          (const float *)input_->deviceMem(), num_inputs_,
                          &beta,
                          (float *)output_->deviceMem(), num_outputs_));
      break;
    case Tensor::Type::HALF:
      chkCuda(cublasHgemm(n.cublas_, CUBLAS_OP_T, CUBLAS_OP_N,
                          num_outputs_, input_->n, num_inputs_,
                          &halpha,
                          (const __half *)weights_->deviceMem(), num_inputs_,
                          (const __half *)input_->deviceMem(), num_inputs_,
                          &hbeta,
                          (__half *)output_->deviceMem(), num_outputs_));
      break;
    default:
      abort();
    }

    chkCUDNN(cudnnAddTensor(n.cudnn_,
                            &alpha, bias_->desc(), bias_->deviceMem(),
                            &alpha, output_->desc(), output_->deviceMem()));
  }


protected:

  Tensor *input_;

  const int num_inputs_;
  const int num_outputs_;

  shared_ptr<Tensor> weights_;
  shared_ptr<Tensor> bias_;

  unique_ptr<Tensor> output_;
};


class FullyConnectedBackProp : public FullyConnected {

public:
  FullyConnectedBackProp(int num_outputs,
                         const Layer &prev,
                         Network &n,
                         const char *weights,
                         const char *bias)
    : FullyConnected(num_outputs, prev, n, weights, bias)
    , input_grad_(prev.gradient())
    , weights_grad_(make_unique<Tensor>(*weights_))
    , bias_grad_(make_unique<Tensor>(*bias_))
    , batch_of_one_(Size(input_->n, 1, 1, 1), input_->type(), 1.0f)
    , output_grad_(make_unique<Tensor>(*output_))
    , weights_optimizer_(n.makeOptimizer(*weights_.get()))
    , bias_optimizer_(n.makeOptimizer(*bias_.get()))
  {
    prev.gradient()->allocate();
    weights_grad_->allocate();
    bias_grad_->allocate();
  }


  void backprop(const Network &n) override {

    float alpha = 1.0f, beta = 0.0f;
    __half halpha = 1.0f, hbeta = 0.0f;

    switch(input_->type()) {

    case Tensor::Type::FLOAT:
      chkCuda(cublasSgemm(n.cublas_, CUBLAS_OP_N, CUBLAS_OP_T,
                          num_inputs_, num_outputs_, input_->n,
                          &alpha,
                          (const float *)input_->deviceMem(), num_inputs_,
                          (const float *)output_grad_->deviceMem(),
                          num_outputs_,
                          &beta,
                          (float *)weights_grad_->deviceMem(), num_inputs_));


      chkCuda(cublasSgemv(n.cublas_, CUBLAS_OP_N, num_outputs_, input_->n,
                          &alpha,
                          (const float *)output_grad_->deviceMem(),
                          num_outputs_,
                          (const float *)batch_of_one_.deviceMem(), 1,
                          &beta,
                          (float *)bias_grad_->deviceMem(), 1));


      if(input_grad_ != NULL) {
        chkCuda(cublasSgemm(n.cublas_, CUBLAS_OP_N, CUBLAS_OP_N,
                            num_inputs_, input_->n, num_outputs_,
                            &alpha,
                            (const float *)weights_->deviceMem(), num_inputs_,
                            (const float *)output_grad_->deviceMem(),
                            num_outputs_,
                            &beta,
                            (float *)input_grad_->deviceMem(), num_inputs_));
      }
      break;

    case Tensor::Type::HALF:
      chkCuda(cublasHgemm(n.cublas_, CUBLAS_OP_N, CUBLAS_OP_T,
                          num_inputs_, num_outputs_, input_->n,
                          &halpha,
                          (const __half *)input_->deviceMem(), num_inputs_,
                          (const __half *)output_grad_->deviceMem(),
                          num_outputs_,
                          &hbeta,
                          (__half *)weights_grad_->deviceMem(), num_inputs_));

      // No cublasSgemv() for half type, so do matrix*matrix instead
      chkCuda(cublasHgemm(n.cublas_, CUBLAS_OP_N, CUBLAS_OP_T,
                          1, num_outputs_, input_->n,
                          &halpha,
                          (const __half *)batch_of_one_.deviceMem(),
                          1,
                          (const __half *)output_grad_->deviceMem(),
                          num_outputs_,
                          &hbeta,
                          (__half *)bias_grad_->deviceMem(), 1));

      if(input_grad_ != NULL) {
        chkCuda(cublasHgemm(n.cublas_, CUBLAS_OP_N, CUBLAS_OP_N,
                            num_inputs_, input_->n, num_outputs_,
                            &halpha,
                            (const __half *)weights_->deviceMem(), num_inputs_,
                            (const __half *)output_grad_->deviceMem(),
                            num_outputs_,
                            &hbeta,
                            (__half *)input_grad_->deviceMem(), num_inputs_));
      }
      break;
    default:
      abort();
    }

    weights_optimizer_->optimize(*weights_.get(), *weights_grad_, n);
    bias_optimizer_->optimize(*bias_.get(), *bias_grad_, n);
  }

  Tensor *gradient() const {
    return output_grad_.get();
  }

protected:
  const Tensor *input_grad_;
  unique_ptr<Tensor> weights_grad_;
  unique_ptr<Tensor> bias_grad_;
  Tensor batch_of_one_;

  unique_ptr<Tensor> output_grad_;

  unique_ptr<Optimizer> weights_optimizer_;
  unique_ptr<Optimizer> bias_optimizer_;

};


std::shared_ptr<Layer> makeFullyConnected(int num_outputs,
                                          const Layer &prev,
                                          Network &n,
                                          const char *weights,
                                          const char *bias)
{
  if(0) {
    // Fully connected layers can be done via convolution layers,
    // but it's significantly slower
    int w = prev.output()->w;
    int h = prev.output()->h;
    assert(w == h);
    return makeConvolution(num_outputs, w, 1, 0, prev, n, true, weights, bias);
  }


  if(n.backprop_)
    return std::make_shared<FullyConnectedBackProp>(num_outputs, prev, n,
                                                    weights, bias);
  else
    return std::make_shared<FullyConnected>(num_outputs, prev, n, weights,
                                            bias);
}


}
