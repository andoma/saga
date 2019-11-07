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
