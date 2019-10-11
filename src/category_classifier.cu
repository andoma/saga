#include <assert.h>
#include <memory>
#include <algorithm>

#include "common.h"

using namespace std;


template< typename T, typename L > __global__ static void
pred(int n, const T *input, L *output, unsigned int channels)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i >= n)
    return;

  input  += channels * i;

  L label = 0;
  T best = input[0];

  for(unsigned int j = 1; j < channels; j++) {
    if(input[j] > best) {
      best = input[j];
      label = j;
    }
  }
  output[i] = label;
}


template< typename T, typename L > __global__ static void
bp(int n, const T *prob, T *grad, const L *labels, float *loss,
   unsigned int channels, T scale)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i >= n)
    return;

  prob += channels * i;
  grad += channels * i;

  L label = labels[i];
  for(unsigned int j = 0; j < channels; j++) {
    grad[j] = (prob[j] - (label == j ? 1.0f : 0.0f)) * scale;
  }
  loss[i] = -log(prob[label]);
}



namespace saga {

class CatClassifier : public Layer {

public:
  CatClassifier(const Layer &prev, Tensor::Type type)
    : input_(prev.output())
    , prob_(*input_)
    , output_(make_unique<Tensor>(Size(prev.output()->n, 1, 1, 1), type))
  {
    prev.output()->allocate();
    prob_.allocate();
  }

  Tensor *output() const override {
    return output_.get();
  }

  std::string name() const override {
    std::stringstream ss;
    ss << "CatClassifier " << input_->name() << " => " << output_->name();
    return ss.str();
  }

  void forward(const Network &n) override {

    float alpha = 1.0f, beta = 0.0f;

    chkCUDNN(cudnnSoftmaxForward(n.cudnn_,
                                 CUDNN_SOFTMAX_ACCURATE,
                                 CUDNN_SOFTMAX_MODE_CHANNEL,
                                 &alpha,
                                 input_->desc(), input_->deviceMem(),
                                 &beta,
                                 prob_.desc(), prob_.deviceMem()));
    const int bs = prob_.n;

    pred<<<(bs+255)/256, 256>>>(bs,
                                (const float *)prob_.deviceMem(),
                                (uint8_t *)output_->deviceMem(),
                                prob_.c);
  }

protected:
  const Tensor *input_;
  Tensor prob_;
  unique_ptr<Tensor> output_;
};


class CatClassifierBackProp : public CatClassifier {
public:
  CatClassifierBackProp(const Layer &prev, Tensor::Type type)
    : CatClassifier(prev, type)
    , input_grad_(prev.gradient())
    , loss_(Size(prev.output()->n, 1, 1, 1), Tensor::Type::FLOAT)
    , labels_(make_unique<Tensor>(Size(prev.output()->n, 1, 1, 1), type))
  {
    prev.gradient()->allocate();
    loss_.allocate();
  }

  void backprop(const Network &n) override {

    int bs = prob_.n;
    float scale = 1.0f / bs;

    bp<<<(bs+255)/256, 256>>>(bs,
                              (const float *)prob_.deviceMem(),
                              (float *)input_grad_->deviceMem(),
                              (const uint8_t *)labels_->deviceMem(),
                              (float *)loss_.deviceMem(),
                              prob_.c, scale);

  }

  Tensor *gradient() const {
    return labels_.get();
  }

  float loss() const override {
    const int bs = loss_.n;
    loss_.synchronize();
    double s = 0;
    for(unsigned int i = 0; i < bs; i++) {
      s += loss_.get(i, 0, 0, 0);
    }
    return s / bs;
  }

protected:
  const Tensor *input_grad_;
  Tensor loss_;
  unique_ptr<Tensor> labels_;
};



std::shared_ptr<Layer> makeCatClassifier(const Layer &prev,
                                         Tensor::Type type,
                                         const Network &n)
{
  if(n.backprop_)
    return std::make_shared<CatClassifierBackProp>(prev, type);
  else
    return std::make_shared<CatClassifier>(prev, type);
}

}

