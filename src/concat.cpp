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


class ConcatTensor : public Tensor {

public:
  ConcatTensor(const Size &s, Type type, const vector<Tensor *> &parts)
    : Tensor(s, type)
    , parts_(parts)
  {}

  ~ConcatTensor()
  {}

  void allocate(cudnnTensorFormat_t format) override;

private:

  const vector<Tensor *> parts_;

};

void ConcatTensor::allocate(cudnnTensorFormat_t format)
{
  Tensor::allocate(format);
  int channels = 0;
  for(size_t i = 0; i < parts_.size(); i++) {
    auto part = parts_[i];
    part->allocate(this, getAddr(0, channels));
    channels += part->c;
  }
}





class Concat : public Layer {

public:
  Concat(const vector<const Layer *> &prevs, bool backprop)
  {
    assert(prevs.size() > 0);
    const Tensor *t0 = prevs[0]->output();

    unsigned int channels = t0->c;
    auto dt = t0->type();
    for(size_t i = 1; i < prevs.size(); i++) {
      channels += prevs[i]->output()->c;
      assert(prevs[i]->output()->w == t0->w);
      assert(prevs[i]->output()->h == t0->h);
      assert(prevs[i]->output()->type() == dt);
    }

    Size s(t0->n, channels, t0->h, t0->w);

    vector<Tensor *> output_parts;
    output_parts.resize(prevs.size());
    transform(prevs.begin(), prevs.end(), output_parts.begin(),
              [](const Layer *l) -> Tensor * { return l->output(); });

    output_ = make_unique<ConcatTensor>(s, dt, output_parts);

    if(backprop) {

      vector<Tensor *> output_grad_parts;
      output_grad_parts.resize(prevs.size());
      transform(prevs.begin(), prevs.end(), output_grad_parts.begin(),
                [](const Layer *l) -> Tensor * { return l->gradient(); });

      output_grad_ = make_unique<ConcatTensor>(s, dt, output_grad_parts);
    }
  }

  Tensor *output() const override {
    return output_.get();
  }

  Tensor *gradient() const override {
    return (Tensor *)output_grad_.get();
  }

  string name() const override {
    stringstream ss;
    ss << "Concat { ... ";
    ss << " } => " << output_->name();
    return ss.str();
  }

  void forward(const Network &n) override {
  }



  void backprop(const Network &n) override {
  }

private:
  unique_ptr<Tensor> output_;
  unique_ptr<Tensor> output_grad_;
};


shared_ptr<Layer> makeConcat(const vector<const Layer *> &inputs,
                             const Network &n)
{
  return make_shared<Concat>(inputs, n.backprop_);
}

}

