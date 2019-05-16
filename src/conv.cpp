#include <memory>
#include <algorithm>

#include "common.h"

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
              const TensorDescriptor &input,
              const InitData &id,
              const Network &n)
    : input_(input)
    , kernel_size_(activation_maps, input.c, filter_size, filter_size)
    , kernel_(TensorDescriptor(input.data_type, kernel_size_))
  {

    const auto data_type = input.data_type;

    kernel_.loadOrRandomize(id, "weights", sqrt(1.0 / (input.c *
                                                       filter_size *
                                                       filter_size)));

    chkCUDNN(cudnnCreateFilterDescriptor(&filter_desc_));
    chkCUDNN(cudnnSetFilter4dDescriptor(filter_desc_,
                                        data_type,
                                        CUDNN_TENSOR_NCHW,
                                        kernel_size_.n,
                                        kernel_size_.c,
                                        kernel_size_.h,
                                        kernel_size_.w));

    chkCUDNN(cudnnCreateConvolutionDescriptor(&conv_desc_));
    chkCUDNN(cudnnSetConvolution2dDescriptor(conv_desc_,
                                             padding, padding,
                                             stride, stride,
                                             1, 1,
                                             CUDNN_CROSS_CORRELATION,
                                             data_type));

    int on, oc, oh, ow;
    chkCUDNN(cudnnGetConvolution2dForwardOutputDim(conv_desc_,
                                                   input.desc(),
                                                   filter_desc_,
                                                   &on, &oc, &oh, &ow));

    output_ = std::make_unique<Tensor>(TensorDescriptor(data_type, Size(on, oc, oh, ow)));

    chkCUDNN(cudnnGetConvolutionForwardAlgorithm(n.cudnn_,
                                                 input.desc(),
                                                 filter_desc_,
                                                 conv_desc_,
                                                 output_->desc(),
                                                 CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                                 0,
                                                 &conv_fwd_algo_));

    size_t workspace_bytes;
    chkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(n.cudnn_,
                                                     input.desc(),
                                                     filter_desc_,
                                                     conv_desc_,
                                                     output_->desc(),
                                                     conv_fwd_algo_,
                                                     &workspace_bytes));

    workspace_size_ = std::max(workspace_size_, workspace_bytes);


    bias_ = std::make_unique<Tensor>(TensorDescriptor(data_type,
                                                      Size(1, oc, 1, 1)));
    bias_->loadOrRandomize(id, "bias", 0);
  }


  ~Convolution() {

  }

  const Tensor *output() const override {
    return output_.get();
  }

  std::string name() const override {
    std::stringstream ss;

    ss << "Convolution " << input_.name()
       << " x " << kernel_.name()
       << " (>" << fwdalgostr(conv_fwd_algo_)
       <<  ") => " << output_->name();
    return ss.str();
  }

  const Tensor *forward(const Network &n,
                        const Tensor &input,
                        bool inference) override {

    float alpha = 1.0f, beta = 0.0f;

    chkCUDNN(cudnnConvolutionForward(n.cudnn_, &alpha,
                                     input.desc(),
                                     input.deviceMem(),
                                     filter_desc_,
                                     kernel_.deviceMem(),
                                     conv_desc_,
                                     conv_fwd_algo_,
                                     n.workspace_, n.workspace_size_,
                                     &beta,
                                     output_->desc(),
                                     output_->deviceMem()));

    chkCUDNN(cudnnAddTensor(n.cudnn_,
                            &alpha, bias_->desc(), bias_->deviceMem(),
                            &alpha, output_->desc(), output_->deviceMem()));

    return output_.get();
  }

protected:

  const TensorDescriptor input_;
  const Size kernel_size_;

  Tensor kernel_;

  std::unique_ptr<Tensor> bias_;

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
                      const TensorDescriptor &input,
                      const InitData &id,
                      const Network &n)
    : Convolution(activation_maps, filter_size,
                  stride, padding, input, id, n)
    , input_grad_(input)
    , kernel_grad_(TensorDescriptor(kernel_))
    , bias_grad_(TensorDescriptor(*bias_))
    , kernel_optimizer_(n.makeOptimizer(kernel_))
    , bias_optimizer_(n.makeOptimizer(*bias_))
  {
    chkCUDNN(cudnnGetConvolutionBackwardDataAlgorithm(n.cudnn_,
                                                      filter_desc_,
                                                      output_->desc(),
                                                      conv_desc_,
                                                      input_grad_.desc(),
                                                      CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
                                                      0,
                                                      &bwd_data_algo_));


    size_t workspace_bytes = 0;

    chkCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(n.cudnn_,
                                                          filter_desc_,
                                                          output_->desc(),
                                                          conv_desc_,
                                                          input_grad_.desc(),
                                                          bwd_data_algo_,
                                                          &workspace_bytes));

    workspace_size_ = std::max(workspace_size_, workspace_bytes);

    chkCUDNN(cudnnGetConvolutionBackwardFilterAlgorithm(n.cudnn_,
                                                        input_grad_.desc(),
                                                        output_->desc(),
                                                        conv_desc_,
                                                        filter_desc_,
                                                        CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,
                                                        0,
                                                        &bwd_filter_algo_));

    chkCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(n.cudnn_,
                                                            input_grad_.desc(),
                                                            output_->desc(),
                                                            conv_desc_,
                                                            filter_desc_,
                                                            bwd_filter_algo_,
                                                            &workspace_bytes));

    workspace_size_ = std::max(workspace_size_, workspace_bytes);
  }



  const Tensor *backprop(const Network &n,
                         const Tensor &input,
                         const Tensor &dy,
                         unsigned int iteration) override {

    float alpha = 1.0f, beta = 0.0f;

    chkCUDNN(cudnnConvolutionBackwardBias(n.cudnn_, &alpha,
                                          dy.desc(), dy.deviceMem(),
                                          &beta,
                                          bias_grad_.desc(),
                                          bias_grad_.deviceMem()));

    chkCUDNN(cudnnConvolutionBackwardFilter(n.cudnn_, &alpha,
                                            input.desc(),
                                            input.deviceMem(),
                                            dy.desc(), dy.deviceMem(),
                                            conv_desc_,
                                            bwd_filter_algo_,
                                            n.workspace_, n.workspace_size_,
                                            &beta,
                                            filter_desc_,
                                            kernel_grad_.deviceMem()));

    chkCUDNN(cudnnConvolutionBackwardData(n.cudnn_, &alpha,
                                          filter_desc_,
                                          kernel_.deviceMem(),
                                          dy.desc(), dy.deviceMem(),
                                          conv_desc_,
                                          bwd_data_algo_,
                                          n.workspace_, n.workspace_size_,
                                          &beta,
                                          input_grad_.desc(),
                                          input_grad_.deviceMem()));

    kernel_optimizer_->optimize(kernel_, kernel_grad_, n, iteration);
    bias_optimizer_->optimize(*bias_, bias_grad_, n, iteration);

    return &input_grad_;
  }


private:
  cudnnConvolutionBwdDataAlgo_t bwd_data_algo_;
  cudnnConvolutionBwdFilterAlgo_t bwd_filter_algo_;

  const Tensor input_grad_;
  const Tensor kernel_grad_;
  const Tensor bias_grad_;


  std::unique_ptr<Optimizer> kernel_optimizer_;
  std::unique_ptr<Optimizer> bias_optimizer_;
};




std::shared_ptr<Layer> makeConvolution(int activation_maps,
                                       int filter_size,
                                       int stride,
                                       int padding,
                                       const TensorDescriptor &input,
                                       const InitData &id,
                                       const Network &n)

{
  if(n.backprop_) {
    return std::make_shared<ConvolutionBackProp>(activation_maps, filter_size,
                                                 stride, padding, input,
                                                 id, n);
  } else {
    return std::make_shared<Convolution>(activation_maps, filter_size,
                                         stride, padding, input, id, n);
  }
}


}
