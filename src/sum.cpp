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

class Sum : public Layer {

public:
  Sum(const vector<const Layer *> &prevs, bool backprop)
  {
    inputs_ = prevs;

    assert(prevs.size() == 2);
    const Tensor *t0 = prevs[0]->output();
    auto dt = t0->type();

    prevs[0]->output()->allocate();
    prevs[1]->output()->allocate();

    for(size_t i = 1; i < prevs.size(); i++) {
      assert((Size)*prevs[i]->output() == (Size)*t0);
    }

    output_ = make_unique<Tensor>(*t0, dt);

    if(backprop) {
      output_grad_ = make_unique<Tensor>(*t0, dt);
    }

    cudnnCreateOpTensorDescriptor(&opdesc_);
    cudnnSetOpTensorDescriptor(opdesc_,
                               CUDNN_OP_TENSOR_ADD,
                               CUDNN_DATA_FLOAT,
                               CUDNN_PROPAGATE_NAN);
  }

  Tensor *output() const override {
    return output_.get();
  }

  Tensor *gradient() const override {
    return (Tensor *)output_grad_.get();
  }

  string name() const override {
    stringstream ss;
    ss << "Sum { ... ";
    ss << " } => " << output_->name();
    return ss.str();
  }

  void forward(const Network &n) override {
    float alpha = 1.0f;
    float beta = 0.0f;

    cudnnOpTensor(n.cudnn_, opdesc_,
                  &alpha,
                  inputs_[0]->output()->desc(),
                  inputs_[0]->output()->deviceMem(),
                  &alpha,
                  inputs_[1]->output()->desc(),
                  inputs_[1]->output()->deviceMem(),
                  &beta,
                  output_->desc(),
                  output_->deviceMem());
  }

  void backprop(const Network &n) override {
    abort(); // Not implemented yet
  }

private:
  unique_ptr<Tensor> output_;
  unique_ptr<Tensor> output_grad_;

  vector<const Layer *> inputs_;
  cudnnOpTensorDescriptor_t opdesc_;

};


shared_ptr<Layer> makeSum(const vector<const Layer *> &inputs,
                          const Network &n)
{
  return make_shared<Sum>(inputs, n.backprop_);
}

}

