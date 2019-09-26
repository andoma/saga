#include <memory>
#include <algorithm>

#include "common.h"

using namespace std;

namespace saga {

static const char *
fwdalgostr(cudnnConvolutionFwdAlgo_t algo)
{
  switch(algo) {
  case CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM:
    return "ImplicitGemm";
  case CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM:
    return "ImplicitPrecompGemm";
  case CUDNN_CONVOLUTION_FWD_ALGO_GEMM:
    return "AlogGem";
  case CUDNN_CONVOLUTION_FWD_ALGO_DIRECT:
    return "Direct";
  case CUDNN_CONVOLUTION_FWD_ALGO_FFT:
    return "FFT";
  case CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING:
    return "FFT-Tiling";
  case CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD:
    return "Winograd";
  case CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED:
    return "Winograd-Nonfused";
  default:
    return "?";
  }
}








class Convolution : public Layer {

public:
  Convolution(int activation_maps,
              int filter_size,
              int stride,
              int padding,
              const Layer &prev,
              const Network &n,
              bool use_bias,
              std::shared_ptr<Tensor> kernel,
              std::shared_ptr<Tensor> bias)
    : input_(prev.output())
    , conv_fwd_algo_(CUDNN_CONVOLUTION_FWD_ALGO_COUNT)
  {
    prev.output()->allocate();
    const auto data_type = input_->dataType();

    if(kernel != NULL) {
      kernel_ = kernel;
    } else {
      kernel_ = std::make_unique<Tensor>(Size(activation_maps,
                                              input_->c,
                                              filter_size,
                                              filter_size),
                                         input_->dataType());

      kernel_->allocate(CUDNN_TENSOR_NHWC);
      kernel_->randomize(sqrt(2.0 / (input_->c * filter_size * filter_size)));
    }

    chkCUDNN(cudnnCreateFilterDescriptor(&filter_desc_));

    chkCUDNN(cudnnSetFilter4dDescriptor(filter_desc_,
                                        kernel_->dataType(),
                                        CUDNN_TENSOR_NHWC,
                                        kernel_->n,
                                        kernel_->c,
                                        kernel_->h,
                                        kernel_->w));

    chkCUDNN(cudnnCreateConvolutionDescriptor(&conv_desc_));
    chkCUDNN(cudnnSetConvolution2dDescriptor(conv_desc_,
                                             padding, padding,
                                             stride, stride,
                                             1, 1,
                                             CUDNN_CROSS_CORRELATION,
                                             data_type));

    int on, oc, oh, ow;
    chkCUDNN(cudnnGetConvolution2dForwardOutputDim(conv_desc_,
                                                   input_->desc(),
                                                   filter_desc_,
                                                   &on, &oc, &oh, &ow));

    if(use_bias) {
      bias_ = bias ?: make_shared<Tensor>(Size(1, oc, 1, 1), input_->dataType(),
                                          0.0f);
    }

    output_ = std::make_unique<Tensor>(Size(on, oc, oh, ow),
                                       input_->dataType());
  }


  void setup(const Network &n) override {

    chkCUDNN(cudnnGetConvolutionForwardAlgorithm(n.cudnn_,
                                                 input_->desc(),
                                                 filter_desc_,
                                                 conv_desc_,
                                                 output_->desc(),
                                                 CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                                 0,
                                                 &conv_fwd_algo_));

    size_t workspace_bytes;
    chkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(n.cudnn_,
                                                     input_->desc(),
                                                     filter_desc_,
                                                     conv_desc_,
                                                     output_->desc(),
                                                     conv_fwd_algo_,
                                                     &workspace_bytes));

    workspace_size_ = std::max(workspace_size_, workspace_bytes);
  }


  ~Convolution() {

  }

  Tensor *output() const override {
    return output_.get();
  }

  std::string name() const override {
    std::stringstream ss;

    ss << "Convolution " << input_->name()
       << " x " << kernel_->name()
       << " (>" << fwdalgostr(conv_fwd_algo_)
       <<  ") => " << output_->name();
    return ss.str();
  }

  void forward(const Network &n) override {

    float alpha = 1.0f, beta = 0.0f;

    chkCUDNN(cudnnConvolutionForward(n.cudnn_, &alpha,
                                     input_->desc(),
                                     input_->deviceMem(),
                                     filter_desc_,
                                     kernel_->deviceMem(),
                                     conv_desc_,
                                     conv_fwd_algo_,
                                     n.workspace_, n.workspace_size_,
                                     &beta,
                                     output_->desc(),
                                     output_->deviceMem()));

    if(bias_) {
      chkCUDNN(cudnnAddTensor(n.cudnn_,
                              &alpha, bias_->desc(), bias_->deviceMem(),
                              &alpha, output_->desc(), output_->deviceMem()));
    }
  }

protected:

  const Tensor *input_;

  std::shared_ptr<Tensor> kernel_;
  std::shared_ptr<Tensor> bias_;

  std::unique_ptr<Tensor> output_;

  cudnnConvolutionDescriptor_t conv_desc_;
  cudnnFilterDescriptor_t filter_desc_;

  cudnnConvolutionFwdAlgo_t conv_fwd_algo_;
};





class ConvolutionBackProp : public Convolution {
public:
  ConvolutionBackProp(int activation_maps,
                      int filter_size,
                      int stride,
                      int padding,
                      const Layer &prev,
                      const Network &n,
                      bool use_bias,
                      std::shared_ptr<Tensor> kernel,
                      std::shared_ptr<Tensor> bias)
    : Convolution(activation_maps, filter_size, stride, padding, prev, n,
                  use_bias, kernel, bias)
    , input_grad_(prev.gradient())
    , output_grad_(*output_)

    , kernel_grad_(make_unique<Tensor>(*kernel_.get()))
    , kernel_optimizer_(n.makeOptimizer(*kernel_))
  {
    if(prev.gradient())
      prev.gradient()->allocate();

    kernel_grad_->allocate();

    if(bias_) {
      bias_grad_ = std::make_unique<Tensor>(*bias_);
      bias_grad_->allocate();
      bias_optimizer_ = n.makeOptimizer(*bias_);
    }
  }


  void setup(const Network &n) override {

    Convolution::setup(n);

    chkCUDNN(cudnnGetConvolutionBackwardDataAlgorithm(n.cudnn_,
                                                      filter_desc_,
                                                      output_->desc(),
                                                      conv_desc_,
                                                      input_->desc(),
                                                      CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
                                                      0,
                                                      &bwd_data_algo_));

    size_t workspace_bytes = 0;

    chkCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(n.cudnn_,
                                                          filter_desc_,
                                                          output_->desc(),
                                                          conv_desc_,
                                                          input_->desc(),
                                                          bwd_data_algo_,
                                                          &workspace_bytes));

    workspace_size_ = std::max(workspace_size_, workspace_bytes);

    chkCUDNN(cudnnGetConvolutionBackwardFilterAlgorithm(n.cudnn_,
                                                        input_->desc(),
                                                        output_->desc(),
                                                        conv_desc_,
                                                        filter_desc_,
                                                        CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,
                                                        0,
                                                        &bwd_filter_algo_));

    chkCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(n.cudnn_,
                                                            input_->desc(),
                                                            output_->desc(),
                                                            conv_desc_,
                                                            filter_desc_,
                                                            bwd_filter_algo_,
                                                            &workspace_bytes));

    workspace_size_ = std::max(workspace_size_, workspace_bytes);
  }



  void backprop(const Network &n) override {

    float alpha = 1.0f, beta = 0.0f;

    if(bias_grad_) {
      chkCUDNN(cudnnConvolutionBackwardBias(n.cudnn_, &alpha,
                                            output_grad_.desc(),
                                            output_grad_.deviceMem(),
                                            &beta,
                                            bias_grad_->desc(),
                                            bias_grad_->deviceMem()));
    }

    chkCUDNN(cudnnConvolutionBackwardFilter(n.cudnn_, &alpha,
                                            input_->desc(),
                                            input_->deviceMem(),
                                            output_grad_.desc(),
                                            output_grad_.deviceMem(),
                                            conv_desc_,
                                            bwd_filter_algo_,
                                            n.workspace_, n.workspace_size_,
                                            &beta,
                                            filter_desc_,
                                            kernel_grad_->deviceMem()));

    if(input_grad_ != NULL) {
      chkCUDNN(cudnnConvolutionBackwardData(n.cudnn_, &alpha,
                                            filter_desc_,
                                            kernel_->deviceMem(),
                                            output_grad_.desc(),
                                            output_grad_.deviceMem(),
                                            conv_desc_,
                                            bwd_data_algo_,
                                            n.workspace_, n.workspace_size_,
                                            &beta,
                                            input_grad_->desc(),
                                            input_grad_->deviceMem()));
    }

    kernel_optimizer_->optimize(*kernel_, *kernel_grad_, n);
    if(bias_optimizer_ != NULL)
      bias_optimizer_->optimize(*bias_, *bias_grad_, n);
  }


  Tensor *gradient() const {
    return (Tensor *)&output_grad_;
  }

private:
  cudnnConvolutionBwdDataAlgo_t bwd_data_algo_;
  cudnnConvolutionBwdFilterAlgo_t bwd_filter_algo_;

  const Tensor *input_grad_;
  const Tensor output_grad_;

  unique_ptr<Tensor> kernel_grad_;

  std::unique_ptr<Tensor> bias_grad_;

  std::unique_ptr<Optimizer> kernel_optimizer_;
  std::unique_ptr<Optimizer> bias_optimizer_;
};




std::shared_ptr<Layer> makeConvolution(int activation_maps,
                                       int filter_size,
                                       int stride,
                                       int padding,
                                       const Layer &prev,
                                       const Network &n,
                                       bool use_bias,
                                       std::shared_ptr<Tensor> kernel,
                                       std::shared_ptr<Tensor> bias)
{
  if(n.backprop_) {
    return std::make_shared<ConvolutionBackProp>(activation_maps, filter_size,
                                                 stride, padding, prev,
                                                 n, use_bias, kernel, bias);
  } else {
    return std::make_shared<Convolution>(activation_maps, filter_size,
                                         stride, padding, prev, n,
                                         use_bias, kernel, bias);
  }
}


}
