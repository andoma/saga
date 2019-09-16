#include <assert.h>

#include <memory>
#include <algorithm>

#include "common.h"

namespace saga {


struct Input {

  Input& operator=(Input const&) = delete;
  Input(Input const &t) = delete;

  Input(const Tensor *input,
        const Tensor *input_grad,
        cudnnTensorDescriptor_t desc,
        void *output_device_mem,
        void *output_grad_device_mem)
    : input_(input)
    , input_grad_(input_grad)
    , desc_(desc)
    , output_device_mem_(output_device_mem)
    , output_grad_device_mem_(output_grad_device_mem)
  {}

  const Tensor *input_;
  const Tensor *input_grad_;
  cudnnTensorDescriptor_t desc_;
  void *output_device_mem_;
  void *output_grad_device_mem_;
};


class Concat : public Layer {

public:
  Concat(const std::vector<const Layer *> &prevs, bool backprop)
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

    output_ = std::make_unique<Tensor>(s, dt);

    cudnnDataType_t odt;
    int on, oc, oh, ow, osn, osc, osh, osw;
    chkCUDNN(cudnnGetTensor4dDescriptor(output_->desc(),
                                        &odt,
                                        &on, &oc, &oh, &ow,
                                        &osn, &osc, &osh, &osw));

    if(backprop)
      output_grad_ = std::make_unique<Tensor>(s, dt);

    char *odm = (char *)output_->deviceMem();
    char *ogdm = backprop ? (char *)output_grad_->deviceMem() : NULL;
    size_t dtsize = sizeof(float);

    for(size_t i = 0; i < prevs.size(); i++) {

      cudnnTensorDescriptor_t desc;
      const Tensor *input = prevs[i]->output();
      cudnnDataType_t dt;
      int n, c, h, w, sn, sc, sh, sw;

      chkCUDNN(cudnnGetTensor4dDescriptor(input->desc(),
                                          &dt,
                                          &n, &c, &h, &w,
                                          &sn, &sc, &sh, &sw));

      chkCUDNN(cudnnCreateTensorDescriptor(&desc));
      chkCUDNN(cudnnSetTensor4dDescriptorEx(desc, dt,
                                            n, c, h, w,
                                            osn, osc, sh, sw));

      inputs_.push_back(std::make_unique<Input>(prevs[i]->output(),
                                                prevs[i]->gradient(),
                                                desc,
                                                (float *)odm,
                                                (float *)ogdm));

      odm += dtsize * sn;
      if(ogdm != NULL)
        ogdm += dtsize * sn;
    }
  }

  const Tensor *output() const override {
    return output_.get();
  }

  Tensor *gradient() const override {
    return (Tensor *)output_grad_.get();
  }

  std::string name() const override {
    std::stringstream ss;
    ss << "Concat {";
    for(const auto &i : inputs_) {
      ss << " " << i->input_->name();
    }
    ss << " } => " << output_->name();
    return ss.str();
  }

  void forward(const Network &n) override {
    float alpha = 1.0f, beta = 0.0f;
    for(const auto &it : inputs_) {
      chkCUDNN(cudnnTransformTensor(n.cudnn_,
                                    &alpha,
                                    it->input_->desc(),
                                    it->input_->deviceMem(),
                                    &beta,
                                    it->desc_,
                                    it->output_device_mem_));
    }

  }



  void backprop(const Network &n) override {
    float alpha = 1.0f, beta = 0.0f;
    for(const auto &it : inputs_) {
      chkCUDNN(cudnnTransformTensor(n.cudnn_,
                                    &alpha,
                                    it->desc_,
                                    it->output_grad_device_mem_,
                                    &beta,
                                    it->input_grad_->desc(),
                                    it->input_grad_->deviceMem()));
    }
  }

protected:
  std::vector<std::unique_ptr<Input>> inputs_;
  std::unique_ptr<Tensor> output_;
  std::unique_ptr<Tensor> output_grad_;
};


std::shared_ptr<Layer> makeConcat(const std::vector<const Layer *> &prevs,
                                  const Network &n)
{
  return std::make_shared<Concat>(prevs, n.backprop_);
}

}

