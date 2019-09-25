#include <assert.h>

#include <memory>
#include <algorithm>

#include "common.h"


using namespace std;

namespace saga {


class ConcatTensor : public Tensor {

public:
  ConcatTensor(const Size &s, cudnnDataType_t dt, const vector<Tensor *> &parts)
    : Tensor(s, dt)
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
  size_t datasize = sizeof(float);

  for(size_t i = 0; i < parts_.size(); i++) {
    auto part = parts_[i];
    const size_t offset = datasize * channels * cs_;
    part->allocate(this, offset);
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
    auto dt = t0->dataType();
    assert(dt == CUDNN_DATA_FLOAT);
    for(size_t i = 1; i < prevs.size(); i++) {
      channels += prevs[i]->output()->c;
      assert(prevs[i]->output()->w == t0->w);
      assert(prevs[i]->output()->h == t0->h);
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

