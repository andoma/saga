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

#include <assert.h>
#include <memory>
#include <algorithm>

#include "common.h"

using namespace std;

namespace saga {

class DropoutBackProp : public Layer {

public:
  DropoutBackProp(float prob,
                  const Layer &prev,
                  const Network &n)
    : input_(prev.output())
    , input_grad_(prev.gradient())
    , output_(make_unique<Tensor>(*input_))
    , output_grad_(make_unique<Tensor>(*input_))
  {
    prev.output()->allocate();
    prev.gradient()->allocate();

    chkCUDNN(cudnnDropoutGetReserveSpaceSize(input_->desc(), &reserve_size_));

    chkCuda(cudaMalloc(&reserve_, reserve_size_));

    chkCUDNN(cudnnDropoutGetStatesSize(n.cudnn_, &states_size_));
    chkCuda(cudaMalloc(&states_, states_size_));

    chkCUDNN(cudnnCreateDropoutDescriptor(&desc_));
    chkCUDNN(cudnnSetDropoutDescriptor(desc_, n.cudnn_, prob,
                                       states_, states_size_, 0)); //rand()));
  }

  void forward(const Network &n) override {

    if(n.inference_) {
      float alpha = 1.0f, beta = 0.0f;

      chkCUDNN(cudnnTransformTensor(n.cudnn_,
                                    &alpha,
                                    input_->desc(), input_->deviceMem(),
                                    &beta,
                                    output_->desc(), output_->deviceMem()));
      return;
    }

    chkCUDNN(cudnnDropoutForward(n.cudnn_, desc_,
                                 input_->desc(), input_->deviceMem(),
                                 output_->desc(), output_->deviceMem(),
                                 reserve_, reserve_size_));
  }

  void backprop(const Network &n) override {

    chkCUDNN(cudnnDropoutBackward(n.cudnn_, desc_,
                                  output_grad_->desc(), output_grad_->deviceMem(),
                                  input_grad_->desc(), input_grad_->deviceMem(),
                                  reserve_, reserve_size_));
  }

  Tensor *output() const override {
    return output_.get();
  }

  Tensor *gradient() const {
    return output_grad_.get();
  }

  std::string name() const override {
    std::stringstream ss;
    ss << "Dropout " << input_->name() << " => " << output_->name();
    return ss.str();
  }

  const Tensor *input_;
  const Tensor *input_grad_;

  unique_ptr<Tensor> output_;
  unique_ptr<Tensor> output_grad_;

  cudnnDropoutDescriptor_t desc_;
  size_t reserve_size_;
  void *reserve_;

  size_t states_size_;
  void *states_;
};



std::shared_ptr<Layer> makeDropout(float prob,
                                   std::shared_ptr<Layer> prev,
                                   const Network &n)
{
  if(n.backprop_) {
    return std::make_shared<DropoutBackProp>(prob, *prev, n);
  } else {
    return nullptr;
  }
}


}

