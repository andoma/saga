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

#include <sstream>
#include <x86intrin.h>
#include "saga.h"
#include "tensor.h"
#include "context.h"

#include "cuda_common.h"
#include "cuda_tensor.h"
#include "cuda_kernels.h"

namespace saga {


struct CudnnOperation : public CudaOperation {

  CudnnOperation(const std::string &name)
    : CudaOperation(name)
  {}

  virtual cudnnStatus_t exec(CudaProgram &p) = 0;

  const char *exec(CudaProgram &p, long batch) {

    cudnnStatus_t s = exec(p);
    if(s == CUDNN_STATUS_SUCCESS)
      return NULL;

    return cudnnGetErrorString(s);
  }
};


//------------------------------------------------------------------------
struct CudnnAdam : public CudaOperation {

  const std::shared_ptr<CudaTensor> weights_, gradient_;
  float learning_rate_;
  float *temp_;
  int iter_;

  CudnnAdam(CudaProgram &p,
            std::shared_ptr<CudaTensor> weights,
            std::shared_ptr<CudaTensor> gradient)
    : CudaOperation("adam")
    , weights_(weights)
    , gradient_(gradient)
    , learning_rate_(p.learning_rate_)
    , iter_(0)
    , ctx_(p.ctx_)
  {
    assert(weights->dims_ == gradient->dims_);

    switch(weights->data_type_) {
    case Tensor::DataType::FLOAT:
      // Allocate 2x floats for each weight (m and v)
      bytes_ = weights_->elements_ * 2 * sizeof(float);
      chkCuda(cudaMallocManaged(&temp_, bytes_));
      chkCuda(cudaMemset(temp_, 0, bytes_));
      break;

    case Tensor::DataType::HALF:
      // Allocate 3x floats for each weight (m and v and float32 copy)
      bytes_ = weights_->elements_ * 3 * sizeof(float);
      chkCuda(cudaMallocManaged(&temp_, bytes_, cudaMemAttachGlobal));
      {
        const uint16_t *src = (const uint16_t *)weights->deviceMem();
        float *dst = temp_;
        for(int i = 0; i < weights->elements_; i++) {
          *dst++ = 0;
          *dst++ = 0;
          *dst++ = _cvtsh_ss(*src++);
        }
      }
      break;

    default:
      break;
    }
  }

  ~CudnnAdam()
  {
    chkCuda(cudaFree(temp_));
  }

  std::vector<std::shared_ptr<CudaTensor>> getInputs() const override {
    return {gradient_, weights_};
  }

  std::vector<std::shared_ptr<CudaTensor>> getOutputs() const override {
    return {weights_};
  }


  const char *exec(CudaProgram &p, long batch) override {
    const int i = ++iter_;

#define ADAM_B1      0.9
#define ADAM_B2      0.999

    const float b1t = 1.0 / (1.0 - pow(ADAM_B1, i));
    const float b2t = 1.0 / (1.0 - pow(ADAM_B2, i));

    switch(weights_->data_type_) {
    case Tensor::DataType::FLOAT:
      adam_float(weights_->elements_,
                 (float *)weights_->deviceMem(),
                 (const float *)gradient_->deviceMem(),
                 (float *)temp_, b1t, b2t, learning_rate_,
                 p.ctx_->stream_);
      break;
    case Tensor::DataType::HALF:
      adam_mixed(weights_->elements_, 1.0f / p.mp_scaling_,
                 (__half *)weights_->deviceMem(),
                 (const __half *)gradient_->deviceMem(),
                 (float *)temp_, b1t, b2t, learning_rate_,
                 (int *)p.check_result_,
                 p.ctx_->stream_);
      break;
    default:
      return "Unsupported tensor datatype";
    }
    return NULL;
  }
  size_t bytes_;
  const std::shared_ptr<CudaContext> ctx_;
};



//------------------------------------------------------------------------

struct CudnnAddTensor : public CudnnOperation {

  const std::shared_ptr<CudaContext> ctx_;
  const std::shared_ptr<CudaTensor> x_, y_;

  CudnnAddTensor(std::shared_ptr<CudaContext> ctx,
                 std::shared_ptr<CudaTensor> x,
                 std::shared_ptr<CudaTensor> y)
    : CudnnOperation("add")
    , ctx_(ctx), x_(x), y_(y)
  {}

  cudnnStatus_t exec(CudaProgram &p)
  {
    float alpha = 1.0f;
    return cudnnAddTensor(ctx_->cudnn_,
                          &alpha, x_->desc(), x_->deviceMem(),
                          &alpha, y_->desc(), y_->deviceMem());
  }

  std::vector<std::shared_ptr<CudaTensor>> getInputs() const override {
    return {x_, y_};
  }

  std::vector<std::shared_ptr<CudaTensor>> getOutputs() const override {
    return {y_};
  }

};






//------------------------------------------------------------------------

struct CudnnTransform : public CudnnOperation {

  const std::shared_ptr<CudaTensor> a_, b_;
  const float beta_;

  CudnnTransform(std::shared_ptr<CudaTensor> a,
                 std::shared_ptr<CudaTensor> b,
                 float beta)
    : CudnnOperation("transform")
    , a_(a), b_(b), beta_(beta)
  {}

  cudnnStatus_t exec(CudaProgram &p) {
    float alpha = 1.0f;
    return cudnnTransformTensor(p.ctx_->cudnn_,
                                &alpha,
                                a_->desc(),
                                a_->deviceMem(),
                                &beta_,
                                b_->desc(),
                                b_->deviceMem());
  }

  std::vector<std::shared_ptr<CudaTensor>> getInputs() const override {
    return {a_};
  }

  std::vector<std::shared_ptr<CudaTensor>> getOutputs() const override {
    return {b_};
  }

};

//------------------------------------------------------------------------


static const char *
convfwdalgostr(cudnnConvolutionFwdAlgo_t algo)
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

static const char *
convbwddataalgostr(cudnnConvolutionBwdDataAlgo_t algo)
{
  switch(algo) {
  case CUDNN_CONVOLUTION_BWD_DATA_ALGO_0:
    return "Algo_0";
  case CUDNN_CONVOLUTION_BWD_DATA_ALGO_1:
    return "Algo_1";
  case CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT:
    return "FFT";
  case CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING:
    return "FFT-Tiling";
  case CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD:
    return "Winograd";
  case CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED:
    return "Winograd-Nonfused";
  default:
    return "?";
  }
}

static const char *
convbwdfilteralgostr(cudnnConvolutionBwdFilterAlgo_t algo)
{
  switch(algo) {
  case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0:
    return "Algo_0";
  case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1:
    return "Algo_1";
  case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT:
    return "FFT";
  case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3:
    return "Algo_3";
  case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD:
    return "Winograd";
  case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED:
    return "Winograd-Nonfused";
  case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING:
    return "FFT-Tiling";
  default:
    return "?";
  }
}

struct CudnnConvolutionDesc {

  cudnnConvolutionDescriptor_t conv_desc_;
  cudnnFilterDescriptor_t filter_desc_;
  cudnnConvolutionFwdAlgo_t conv_fwd_algo_;

  cudnnConvolutionBwdDataAlgo_t bwd_data_algo_;
  cudnnConvolutionBwdFilterAlgo_t bwd_filter_algo_;

  CudnnConvolutionDesc()
  {
    chkCUDNN(cudnnCreateFilterDescriptor(&filter_desc_));
    chkCUDNN(cudnnCreateConvolutionDescriptor(&conv_desc_));
  }

  ~CudnnConvolutionDesc()
  {
    chkCUDNN(cudnnDestroyFilterDescriptor(filter_desc_));
    chkCUDNN(cudnnDestroyConvolutionDescriptor(conv_desc_));
  }


  const char *setup(CudaProgram &p,
                    CudaTensor &x,
                    CudaTensor &w,
                    CudaTensor &y,
                    int pad,
                    int stride,
                    bool bwd) {
    cudnnStatus_t s;
    s = cudnnSetFilter4dDescriptor(filter_desc_,
                                   x.type_,
                                   p.tensorFormat(x.data_type_),
                                   w.dims_[0],
                                   w.dims_[1],
                                   w.dims_[2],
                                   w.dims_[3]);
    if(s)
      return cudnnGetErrorString(s);

    s = cudnnSetConvolutionMathType(conv_desc_,
                                    CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION);
    if(s)
      return cudnnGetErrorString(s);

    s = cudnnSetConvolution2dDescriptor(conv_desc_,
                                        pad, pad,
                                        stride, stride,
                                        1, 1,
                                        CUDNN_CROSS_CORRELATION,
                                        CUDNN_DATA_FLOAT);
    if(s)
      return cudnnGetErrorString(s);

    s = cudnnGetConvolutionForwardAlgorithm(p.ctx_->cudnn_,
                                            x.desc_,
                                            filter_desc_,
                                            conv_desc_,
                                            y.desc_,
                                            CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                            0,
                                            &conv_fwd_algo_);
    if(s)
      return cudnnGetErrorString(s);

    size_t workspace;
    s = cudnnGetConvolutionForwardWorkspaceSize(p.ctx_->cudnn_,
                                                x.desc_,
                                                filter_desc_,
                                                conv_desc_,
                                                y.desc_,
                                                conv_fwd_algo_,
                                                &workspace);
    if(s)
      return cudnnGetErrorString(s);

    p.workspace_.request(workspace);

    if(!bwd)
      return NULL;


    s = cudnnGetConvolutionBackwardDataAlgorithm(p.ctx_->cudnn_,
                                                 filter_desc_,
                                                 y.desc(),
                                                 conv_desc_,
                                                 x.desc(),
                                                 CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
                                                 0,
                                                 &bwd_data_algo_);
    if(s)
      return cudnnGetErrorString(s);

    s = cudnnGetConvolutionBackwardDataWorkspaceSize(p.ctx_->cudnn_,
                                                     filter_desc_,
                                                     y.desc(),
                                                     conv_desc_,
                                                     x.desc(),
                                                     bwd_data_algo_,
                                                     &workspace);
    if(s)
      return cudnnGetErrorString(s);

    p.workspace_.request(workspace);

    s = cudnnGetConvolutionBackwardFilterAlgorithm(p.ctx_->cudnn_,
                                                   x.desc(),
                                                   y.desc(),
                                                   conv_desc_,
                                                   filter_desc_,
                                                   CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,
                                                   0,
                                                   &bwd_filter_algo_);
    if(s)
      return cudnnGetErrorString(s);

    s = cudnnGetConvolutionBackwardFilterWorkspaceSize(p.ctx_->cudnn_,
                                                       x.desc(),
                                                       y.desc(),
                                                       conv_desc_,
                                                       filter_desc_,
                                                       bwd_filter_algo_,
                                                       &workspace);
    if(s)
      return cudnnGetErrorString(s);

    p.workspace_.request(workspace);

    return NULL;
  }
};


struct CudnnConvolutionFwd : public CudnnOperation {

  const std::shared_ptr<CudaContext> ctx_;
  const std::shared_ptr<CudnnConvolutionDesc> desc_;
  const std::shared_ptr<CudaTensor> x_, w_, y_;

  CudnnConvolutionFwd(std::shared_ptr<CudaContext> ctx,
                      std::shared_ptr<CudnnConvolutionDesc> desc,
                      std::shared_ptr<CudaTensor> x,
                      std::shared_ptr<CudaTensor> w,
                      std::shared_ptr<CudaTensor> y)
    : CudnnOperation("convfwd")
    , ctx_(ctx), desc_(desc), x_(x), w_(w), y_(y)
  {}

  cudnnStatus_t exec(CudaProgram &p)
  {
    float alpha = 1.0f;
    float beta = 0.0f;

    return cudnnConvolutionForward(ctx_->cudnn_, &alpha,
                                   x_->desc(),
                                   x_->deviceMem(),
                                   desc_->filter_desc_,
                                   w_->deviceMem(),
                                   desc_->conv_desc_,
                                   desc_->conv_fwd_algo_,
                                   p.workspace_.ptr(),
                                   p.workspace_.size(),
                                   &beta,
                                   y_->desc(),
                                   y_->deviceMem());
  }

  std::vector<std::shared_ptr<CudaTensor>> getInputs() const override {
    return {x_, w_};
  }

  std::vector<std::shared_ptr<CudaTensor>> getOutputs() const override {
    return {y_};
  }

  std::string info() const override {
    return convfwdalgostr(desc_->conv_fwd_algo_);
  }

};


struct CudnnConvolutionBwdBias : public CudnnOperation {

  const std::shared_ptr<CudaContext> ctx_;
  const std::shared_ptr<CudaTensor> dy_, db_;

  CudnnConvolutionBwdBias(std::shared_ptr<CudaContext> ctx,
                          std::shared_ptr<CudaTensor> dy,
                          std::shared_ptr<CudaTensor> db)
    : CudnnOperation("convbwdbias")
    , ctx_(ctx), dy_(dy), db_(db)
  {}

  cudnnStatus_t exec(CudaProgram &p)
  {
    float alpha = 1.0f;
    float beta = 0.0f;

    return cudnnConvolutionBackwardBias(ctx_->cudnn_, &alpha,
                                        dy_->desc(),
                                        dy_->deviceMem(),
                                        &beta,
                                        db_->desc(),
                                        db_->deviceMem());
  }

  std::vector<std::shared_ptr<CudaTensor>> getInputs() const override {
    return {dy_};
  }

  std::vector<std::shared_ptr<CudaTensor>> getOutputs() const override {
    return {db_};
  }
};



struct CudnnConvolutionBwdFilter : public CudnnOperation {

  const std::shared_ptr<CudaContext> ctx_;
  const std::shared_ptr<CudnnConvolutionDesc> desc_;
  const std::shared_ptr<CudaTensor> x_, dy_, dw_;

  CudnnConvolutionBwdFilter(std::shared_ptr<CudaContext> ctx,
                            std::shared_ptr<CudnnConvolutionDesc> desc,
                            std::shared_ptr<CudaTensor> x,
                            std::shared_ptr<CudaTensor> dy,
                            std::shared_ptr<CudaTensor> dw)
    : CudnnOperation("convbwdfilter")
    , ctx_(ctx), desc_(desc), x_(x), dy_(dy), dw_(dw)
  {}

  cudnnStatus_t exec(CudaProgram &p)
  {
    float alpha = 1.0f;
    float beta = 0.0f;

    return cudnnConvolutionBackwardFilter(ctx_->cudnn_, &alpha,
                                          x_->desc(),
                                          x_->deviceMem(),
                                          dy_->desc(),
                                          dy_->deviceMem(),
                                          desc_->conv_desc_,
                                          desc_->bwd_filter_algo_,
                                          p.workspace_.ptr(),
                                          p.workspace_.size(),
                                          &beta,
                                          desc_->filter_desc_,
                                          dw_->deviceMem());
  }

  std::vector<std::shared_ptr<CudaTensor>> getInputs() const override {
    return {x_, dy_};
  }

  std::vector<std::shared_ptr<CudaTensor>> getOutputs() const override {
    return {dw_};
  }

  std::string info() const override {
    return convbwdfilteralgostr(desc_->bwd_filter_algo_);
  }

};


struct CudnnConvolutionBwdData : public CudnnOperation {

  const std::shared_ptr<CudaContext> ctx_;
  const std::shared_ptr<CudnnConvolutionDesc> desc_;
  const std::shared_ptr<CudaTensor> w_, dy_, dx_;
  const float dx_beta_;

  CudnnConvolutionBwdData(std::shared_ptr<CudaContext> ctx,
                          std::shared_ptr<CudnnConvolutionDesc> desc,
                          std::shared_ptr<CudaTensor> w,
                          std::shared_ptr<CudaTensor> dy,
                          std::shared_ptr<CudaTensor> dx,
                          float dx_beta)
    : CudnnOperation("convbwddata")
    , ctx_(ctx), desc_(desc), w_(w), dy_(dy), dx_(dx), dx_beta_(dx_beta)
  {}

  cudnnStatus_t exec(CudaProgram &p)
  {
    float alpha = 1.0f;

    return cudnnConvolutionBackwardData(ctx_->cudnn_, &alpha,
                                        desc_->filter_desc_,
                                        w_->deviceMem(),
                                        dy_->desc(),
                                        dy_->deviceMem(),
                                        desc_->conv_desc_,
                                        desc_->bwd_data_algo_,
                                        p.workspace_.ptr(),
                                        p.workspace_.size(),
                                        &dx_beta_,
                                        dx_->desc(),
                                        dx_->deviceMem());

  }

  std::vector<std::shared_ptr<CudaTensor>> getInputs() const override {
    auto r = std::vector{w_, dy_};
    if(dx_beta_)
      r.push_back(dx_);
    return r;
  }

  std::vector<std::shared_ptr<CudaTensor>> getOutputs() const override {
    return {dx_};
  }

  std::string info() const override {
    return convbwddataalgostr(desc_->bwd_data_algo_);
  }
};







static const char *
conv_setup(CudaProgram &p, const Node &n, bool training)
{
  auto x = p.lower_tensor_batch(n.inputs_.get("x"));

  if(x->data_type_ == Tensor::DataType::HALF && !x->cpacked()) {
    assert(!x->grad_);

    auto xx = std::make_shared<CudaTensor>(x->data_type_,
                                           x->dims_,
                                           CUDNN_TENSOR_NHWC,
                                           p.ctx_,
                                           x->namePostfix("nhwc"));

    auto tr = std::make_shared<CudnnTransform>(x, xx, 0.0f);
    if(training)
      p.fwd(tr);
    else
      p.infer(tr);
    x = xx;
  }

  auto y = p.lower_tensor_batch(n.outputs_.get("y"));
  auto w = p.lower_tensor(n.inputs_.get("w"));
  auto b = p.lower_tensor(n.inputs_.get("b"), 2);

  auto desc = std::make_shared<CudnnConvolutionDesc>();

  const int pad = n.attributes_.get("pad", 0);
  const int stride = n.attributes_.get("stride", 1);

  const char *err = desc->setup(p, *x, *w, *y, pad, stride, training);
  if(err)
    return err;

  if(!training) {
    p.infer(std::make_shared<CudnnConvolutionFwd>(p.ctx_, desc, x, w, y));
    if(b)
      p.infer(std::make_shared<CudnnAddTensor>(p.ctx_, b, y));
    return NULL;
  }

  p.fwd(std::make_shared<CudnnConvolutionFwd>(p.ctx_, desc, x, w, y));
  if(b)
    p.fwd(std::make_shared<CudnnAddTensor>(p.ctx_, b, y));

  auto dy = y->makeSharedGrad();

  if(b) {
    auto db = b->makePrivateGrad();
    p.bwd(std::make_shared<CudnnConvolutionBwdBias>(p.ctx_, dy, db));
    p.upd(std::make_shared<CudnnAdam>(p, b, db));
  }

  auto dw = w->makePrivateGrad();
  p.bwd(std::make_shared<CudnnConvolutionBwdFilter>(p.ctx_, desc, x, dy, dw));
  p.upd(std::make_shared<CudnnAdam>(p, w, dw));

  auto dx = x->grad_;
  if(dx) {
    const float dx_beta = n.attributes_.get("dx.beta", 0.0f);

    p.bwd(std::make_shared<CudnnConvolutionBwdData>(p.ctx_, desc, w, dy, dx,
                                                    dx_beta));
  }
  return NULL;
}


REGISTER_CUDA_OP("conv", conv_setup);





//------------------------------------------------------------------------

struct CudnnActivationFwd : public CudnnOperation {
  const std::shared_ptr<CudaTensor> x_, y_;
  const float y_beta_;

  cudnnActivationDescriptor_t desc_;

  CudnnActivationFwd(std::shared_ptr<CudaTensor> x,
                     std::shared_ptr<CudaTensor> y,
                     cudnnActivationMode_t mode,
                     float alpha,
                     float y_beta)
    : CudnnOperation("actfwd")
    , x_(x), y_(y), y_beta_(y_beta)
  {
    chkCUDNN(cudnnCreateActivationDescriptor(&desc_));
    chkCUDNN(cudnnSetActivationDescriptor(desc_, mode,
                                          CUDNN_PROPAGATE_NAN, alpha));
  }

  ~CudnnActivationFwd()
  {
    chkCUDNN(cudnnDestroyActivationDescriptor(desc_));
  }


  cudnnStatus_t exec(CudaProgram &p) {
    float alpha = 1.0f;

    return cudnnActivationForward(p.ctx_->cudnn_, desc_,
                                  &alpha,
                                  x_->desc(), x_->deviceMem(),
                                  &y_beta_,
                                  y_->desc(), y_->deviceMem());
  }

  std::vector<std::shared_ptr<CudaTensor>> getInputs() const override {
    return {x_};
  }

  std::vector<std::shared_ptr<CudaTensor>> getOutputs() const override {
    return {y_};
  }

};


struct CudnnActivationBwd : public CudnnOperation {
  const std::shared_ptr<CudnnActivationFwd> fwd_;
  const std::shared_ptr<CudaTensor> x_, y_, dx_, dy_;
  const float dx_beta_;

  CudnnActivationBwd(const std::shared_ptr<CudnnActivationFwd> fwd,
                     std::shared_ptr<CudaTensor> x,
                     std::shared_ptr<CudaTensor> y,
                     std::shared_ptr<CudaTensor> dx,
                     std::shared_ptr<CudaTensor> dy,
                     float dx_beta)
    : CudnnOperation("actbwd")
    , fwd_(fwd), x_(x), y_(y), dx_(dx), dy_(dy), dx_beta_(dx_beta)
  {}

  cudnnStatus_t exec(CudaProgram &p) {
    float alpha = 1.0f;

    return cudnnActivationBackward(p.ctx_->cudnn_, fwd_->desc_,
                                   &alpha,
                                   y_->desc(), y_->deviceMem(),
                                   dy_->desc(), dy_->deviceMem(),
                                   x_->desc(), x_->deviceMem(),
                                   &dx_beta_,
                                   dx_->desc(), dx_->deviceMem());
  }

  std::vector<std::shared_ptr<CudaTensor>> getInputs() const override {
    auto r = std::vector{x_, y_, dy_};
    if(dx_beta_)
      r.push_back(dx_);
    return r;
  }

  std::vector<std::shared_ptr<CudaTensor>> getOutputs() const override {
    return {dx_};
  }

};



static const char *
activation_setup(CudaProgram &p, const Node &n, bool training,
                 cudnnActivationMode_t mode, float alpha)
{
  auto x = p.lower_tensor_batch(n.inputs_.get("x"));
  auto y = p.lower_tensor_batch(n.outputs_.get("y"));
  auto fwd = std::make_shared<CudnnActivationFwd>(x, y, mode, alpha, 0);

  if(!training) {
    p.infer(fwd);
    return NULL;
  }

  p.fwd(fwd);

  auto dx = x->grad_;
  if(!dx)
    return NULL;
  auto dy = y->makeSharedGrad();

  const float dx_beta = n.attributes_.get("dx.beta", 0.0f);
  p.bwd(std::make_shared<CudnnActivationBwd>(fwd, x, y, dx, dy, dx_beta));
  return NULL;
}



static const char *
relu_setup(CudaProgram &p, const Node &n, bool training)
{
  return activation_setup(p, n, training, CUDNN_ACTIVATION_RELU, 0.0f);
}

REGISTER_CUDA_OP("relu", relu_setup);


static const char *
elu_setup(CudaProgram &p, const Node &n, bool training)
{
  return activation_setup(p, n, training, CUDNN_ACTIVATION_ELU,
                          n.attributes_.get("alpha", 0.1f));
}

REGISTER_CUDA_OP("elu", relu_setup);




//------------------------------------------------------------------------

struct CudnnPoolingFwd : public CudnnOperation {

  const std::shared_ptr<CudaTensor> x_, y_;

  cudnnPoolingDescriptor_t desc_;

  CudnnPoolingFwd(std::shared_ptr<CudaTensor> x,
                  std::shared_ptr<CudaTensor> y,
                  cudnnPoolingMode_t mode,
                  int size,
                  int pad,
                  int stride)
    : CudnnOperation("poolfwd")
    , x_(x), y_(y)
  {
    chkCUDNN(cudnnCreatePoolingDescriptor(&desc_));

    chkCUDNN(cudnnSetPooling2dDescriptor(desc_,
                                         mode,
                                         CUDNN_PROPAGATE_NAN,
                                         size, size,
                                         pad, pad,
                                         stride, stride));
  }

  ~CudnnPoolingFwd()
  {
    chkCUDNN(cudnnDestroyPoolingDescriptor(desc_));
  }

  cudnnStatus_t exec(CudaProgram &p) {
    float alpha = 1.0f;
    float beta = 0.0f;

    return cudnnPoolingForward(p.ctx_->cudnn_, desc_,
                               &alpha,
                               x_->desc(), x_->deviceMem(),
                               &beta,
                               y_->desc(), y_->deviceMem());
  }

  std::vector<std::shared_ptr<CudaTensor>> getInputs() const override {
    return {x_};
  }

  std::vector<std::shared_ptr<CudaTensor>> getOutputs() const override {
    return {y_};
  }

};




struct CudnnPoolingBwd : public CudnnOperation {

  const std::shared_ptr<CudnnPoolingFwd> fwd_;
  const std::shared_ptr<CudaTensor> x_, y_, dx_, dy_;
  const float dx_beta_;

  CudnnPoolingBwd(std::shared_ptr<CudnnPoolingFwd> fwd,
                  std::shared_ptr<CudaTensor> x,
                  std::shared_ptr<CudaTensor> y,
                  std::shared_ptr<CudaTensor> dx,
                  std::shared_ptr<CudaTensor> dy,
                  float dx_beta)
    : CudnnOperation("poolbwd")
    , fwd_(fwd), x_(x), y_(y), dx_(dx), dy_(dy), dx_beta_(dx_beta)
  {}

  cudnnStatus_t exec(CudaProgram &p) {
    float alpha = 1.0f;

    return cudnnPoolingBackward(p.ctx_->cudnn_, fwd_->desc_,
                                &alpha,
                                y_->desc(), y_->deviceMem(),
                                dy_->desc(), dy_->deviceMem(),
                                x_->desc(), x_->deviceMem(),
                                &dx_beta_,
                                dx_->desc(),
                                dx_->deviceMem());
  }

  std::vector<std::shared_ptr<CudaTensor>> getInputs() const override {
    auto r = std::vector{x_, y_, dy_};
    if(dx_beta_)
      r.push_back(dx_);
    return r;
  }

  std::vector<std::shared_ptr<CudaTensor>> getOutputs() const override {
    return {dx_};
  }

};




static const char *
pooling_setup(CudaProgram &p, const Node &n, bool training,
              cudnnPoolingMode_t mode)
{
  auto x = p.lower_tensor_batch(n.inputs_.get("x"));
  auto y = p.lower_tensor_batch(n.outputs_.get("y"));

  int size;
  if(n.attributes_.get("global", false)) {
    size = x->dims_[2];
  } else {
    size = n.attributes_.get("size", 1);
  }
  const int pad    = n.attributes_.get("pad", 0);
  const int stride = n.attributes_.get("stride", 1);

  auto fwd = std::make_shared<CudnnPoolingFwd>(x, y, mode, size, pad, stride);

  if(!training) {
    p.infer(fwd);
    return NULL;
  }

  p.fwd(fwd);

  auto dx = x->grad_;
  if(!dx)
    return NULL;
  auto dy = y->makeSharedGrad();

  const float dx_beta = n.attributes_.get("dx.beta", 0.0f);
  p.bwd(std::make_shared<CudnnPoolingBwd>(fwd, x, y, dx, dy, dx_beta));
  return NULL;
}

static const char *
maxpool_setup(CudaProgram &p, const Node &n, bool training)
{
  return pooling_setup(p, n, training, CUDNN_POOLING_MAX);
}

REGISTER_CUDA_OP("maxpool", maxpool_setup);


static const char *
avgpool_setup(CudaProgram &p, const Node &n, bool training)
{
  return pooling_setup(p, n, training,
                       CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING);
}

REGISTER_CUDA_OP("avgpool", avgpool_setup);



//------------------------------------------------------------------------


static const char *
cublasErrStr(cublasStatus_t error) {
  switch (error) {
  case CUBLAS_STATUS_SUCCESS:
    return NULL;

  case CUBLAS_STATUS_NOT_INITIALIZED:
    return "CUBLAS_STATUS_NOT_INITIALIZED";

  case CUBLAS_STATUS_ALLOC_FAILED:
    return "CUBLAS_STATUS_ALLOC_FAILED";

  case CUBLAS_STATUS_INVALID_VALUE:
    return "CUBLAS_STATUS_INVALID_VALUE";

  case CUBLAS_STATUS_ARCH_MISMATCH:
    return "CUBLAS_STATUS_ARCH_MISMATCH";

  case CUBLAS_STATUS_MAPPING_ERROR:
    return "CUBLAS_STATUS_MAPPING_ERROR";

  case CUBLAS_STATUS_EXECUTION_FAILED:
    return "CUBLAS_STATUS_EXECUTION_FAILED";

  case CUBLAS_STATUS_INTERNAL_ERROR:
    return "CUBLAS_STATUS_INTERNAL_ERROR";

  case CUBLAS_STATUS_NOT_SUPPORTED:
    return "CUBLAS_STATUS_NOT_SUPPORTED";

  case CUBLAS_STATUS_LICENSE_ERROR:
    return "CUBLAS_STATUS_LICENSE_ERROR";
  }

  return "<unknown>";
}



struct CudaGemm : public CudaOperation {

  const cublasOperation_t transa_;
  const cublasOperation_t transb_;
  const int m_;
  const int n_;
  const int k_;
  const std::shared_ptr<CudaTensor> a_;
  const int lda_;
  const std::shared_ptr<CudaTensor> b_;
  const int ldb_;
  const std::shared_ptr<CudaTensor> c_;
  const int ldc_;

  CudaGemm(cublasOperation_t transa,
           cublasOperation_t transb,
           int m, int n, int k,
           std::shared_ptr<CudaTensor> a, int lda,
           std::shared_ptr<CudaTensor> b, int ldb,
           std::shared_ptr<CudaTensor> c, int ldc)
    : CudaOperation("gemm")
    , transa_(transa), transb_(transb), m_(m), n_(n), k_(k),
      a_(a), lda_(lda), b_(b), ldb_(ldb), c_(c), ldc_(ldc)
  {}

  const char *exec(CudaProgram &p, long batch) {
    float alpha = 1.0f, beta = 0.0f;
    __half halpha = 1.0f, hbeta = 0.0f;

    cublasStatus_t s;
    switch(a_->type_) {
    case CUDNN_DATA_FLOAT:

      s = cublasSgemm(p.ctx_->cublas_, transa_, transb_,
                      m_, n_, k_,
                      &alpha,
                      (const float *)a_->deviceMem(), lda_,
                      (const float *)b_->deviceMem(), ldb_,
                      &beta,
                      (float *)c_->deviceMem(), ldc_);
      break;
    case CUDNN_DATA_HALF:
      s = cublasHgemm(p.ctx_->cublas_, transa_, transb_,
                      m_, n_, k_,
                      &halpha,
                      (const __half *)a_->deviceMem(), lda_,
                      (const __half *)b_->deviceMem(), ldb_,
                      &hbeta,
                      (__half *)c_->deviceMem(), ldc_);
      break;
    default:
      return "Unsupported tensor datatype";
    }
    return cublasErrStr(s);
  }

  std::vector<std::shared_ptr<CudaTensor>> getInputs() const override {
    return {a_, b_};
  }

  std::vector<std::shared_ptr<CudaTensor>> getOutputs() const override {
    return {c_};
  }
};


static const char *
fc_setup(CudaProgram &p, const Node &n, bool training)
{
  auto x = p.lower_tensor_batch(n.inputs_.get("x"));
  auto y = p.lower_tensor_batch(n.outputs_.get("y"));

  auto w = p.lower_tensor(n.inputs_.get("w"));
  auto b = p.lower_tensor(n.inputs_.get("b"), 2);

  const bool transW = n.attributes_.get("transW", false);

  const cublasOperation_t transa = transW ? CUBLAS_OP_T : CUBLAS_OP_N;
  const cublasOperation_t transb = CUBLAS_OP_N;

  const int num_inputs  = x->dims_[1];
  const int num_outputs = y->dims_[1];
  const int batch_size  = x->dims_[0];

  auto fwd =
    std::make_shared<CudaGemm>(transa, transb,
                               num_outputs, batch_size, num_inputs,
                               w, transW ? num_inputs : num_outputs,
                               x, num_inputs,
                               y, num_outputs);

  if(!training) {
    p.infer(fwd);
    if(b)
      p.infer(std::make_shared<CudnnAddTensor>(p.ctx_, b, y));
    return NULL;
  }

  p.fwd(fwd);

  if(!transW) {
    // Fix this
    return "fully connected with !transW not suppored";
  }

  auto dw = w->makePrivateGrad();
  auto dy = y->makeSharedGrad();

  p.bwd(std::make_shared<CudaGemm>(CUBLAS_OP_N, CUBLAS_OP_T,
                                   num_inputs, num_outputs, batch_size,
                                   x, num_inputs,
                                   dy, num_outputs,
                                   dw, num_inputs));
  p.upd(std::make_shared<CudnnAdam>(p, w, dw));

  if(b) {
    auto ones =
      p.lower_tensor(Tensor::make(x->data_type_, {batch_size, 1}, 1, 0));

    auto db = b->makePrivateGrad();
    p.bwd(std::make_shared<CudaGemm>(CUBLAS_OP_N, CUBLAS_OP_T,
                                     1, num_outputs, batch_size,
                                     ones, 1,
                                     dy, num_outputs,
                                     db, 1));
    p.upd(std::make_shared<CudnnAdam>(p, b, db));
  }

  auto dx = x->grad_;

  if(dx) {
    const float dx_beta = n.attributes_.get("dx.beta", 0.0f);
    if(dx_beta)
      return "dx_beta != 0";

    p.bwd(std::make_shared<CudaGemm>(CUBLAS_OP_N, CUBLAS_OP_N,
                                     num_inputs, batch_size, num_outputs,
                                     w, num_inputs,
                                     dy, num_outputs,
                                     dx, num_inputs));
  }
  return NULL;
}

REGISTER_CUDA_OP("fc", fc_setup);



//------------------------------------------------------------------------




struct CudaConvert : public CudaOperation {

  const std::shared_ptr<CudaTensor> x_, y_;
  const float scale_;
  void (*algo_)(const void *src, void *dst, int elements, float scale,
                cudaStream_t stream);

  CudaConvert(std::shared_ptr<CudaTensor> x,
              std::shared_ptr<CudaTensor> y,
              float scale)
    : CudaOperation("convert")
    , x_(x)
    , y_(y)
    , scale_(scale)
  {
    if(x_->data_type_ == Tensor::DataType::U8 &&
       y_->data_type_ == Tensor::DataType::FLOAT) {
      algo_ = convert_u8_float;
    } else if(x_->data_type_ == Tensor::DataType::U8 &&
              y_->data_type_ == Tensor::DataType::HALF) {
      algo_ = convert_u8_half;
    } else if(x_->data_type_ == Tensor::DataType::FLOAT &&
              y_->data_type_ == Tensor::DataType::HALF) {
      algo_ = convert_float_half;
    } else {
      abort();
    }
  }

  const char *exec(CudaProgram &p, long batch) {
    algo_(x_->deviceMem(), y_->deviceMem(), x_->elements_, scale_,
          p.ctx_->stream_);
    return NULL;
  }

  std::vector<std::shared_ptr<CudaTensor>> getInputs() const override {
    return {x_};
  }

  std::vector<std::shared_ptr<CudaTensor>> getOutputs() const override {
    return {y_};
  }
};



static const char *
convert_setup(CudaProgram &p, const Node &n, bool training)
{
  auto x = p.lower_tensor_batch(n.inputs_.get("x"));
  auto y = p.lower_tensor_batch(n.outputs_.get("y"), *x);
  auto scale = n.attributes_.get("scale", 1.0f);

  auto op = std::make_shared<CudaConvert>(x, y, scale);

  if(!training) {
    p.infer(op);
  } else {
    p.fwd(op);
  }
  return NULL;
}

REGISTER_CUDA_OP("convert", convert_setup);

//------------------------------------------------------------------------
struct CudaCatClassifierFwd : public CudaOperation {

  const std::shared_ptr<CudaTensor> x_, y_;

  CudaCatClassifierFwd(std::shared_ptr<CudaTensor> x,
                       std::shared_ptr<CudaTensor> y)
    : CudaOperation("catclassifierfwd")
    , x_(x), y_(y)
  {}

  const char *exec(CudaProgram &p, long batch) {
    switch(x_->type_) {
    case CUDNN_DATA_FLOAT:
      catclassifier_fwd_float_i32(x_->dims_[0],
                                  (const float *)x_->deviceMem(),
                                  (int32_t *)y_->deviceMem(), x_->dims_[1],
                                  p.ctx_->stream_);
      break;
    case CUDNN_DATA_HALF:
      catclassifier_fwd_half_i32(x_->dims_[0],
                                 (const __half *)x_->deviceMem(),
                                 (int32_t *)y_->deviceMem(), x_->dims_[1],
                                 p.ctx_->stream_);
      break;
    default:
      return "Unsupported tensor datatype";
    }
    return NULL;
  }

  std::vector<std::shared_ptr<CudaTensor>> getInputs() const override {
    return {x_};
  }

  std::vector<std::shared_ptr<CudaTensor>> getOutputs() const override {
    return {y_};
  }

};

struct CudaCatClassifierBwd : public CudaOperation {

  const std::shared_ptr<CudaTensor> x_, y_, dx_, dy_, loss_;

  CudaCatClassifierBwd(std::shared_ptr<CudaTensor> x,
                       std::shared_ptr<CudaTensor> y,
                       std::shared_ptr<CudaTensor> dx,
                       std::shared_ptr<CudaTensor> dy,
                       std::shared_ptr<CudaTensor> loss)
    : CudaOperation("catclassifierbwd")
    , x_(x), y_(y), dx_(dx), dy_(dy), loss_(loss)
  {}

  const char *exec(CudaProgram &p, long batch) {

    const int n = x_->dims_[0];
    const int c = x_->dims_[1];
    const float scale = 1.0f / n;

    switch(x_->type_) {
    case CUDNN_DATA_FLOAT:
      catclassifier_bwd_float_i32(n,
                                  (const float *)x_->deviceMem(),
                                  (float *)dx_->deviceMem(),
                                  (const int32_t *)y_->deviceMem(),
                                  (const int32_t *)dy_->deviceMem(),
                                  loss_ ? (float *)loss_->deviceMem() : NULL,
                                  c, scale,
                                  p.ctx_->stream_);

      break;
    case CUDNN_DATA_HALF:
      catclassifier_bwd_half_i32(n,
                                 (const __half *)x_->deviceMem(),
                                 (__half *)dx_->deviceMem(),
                                 (const int32_t *)y_->deviceMem(),
                                 (const int32_t *)dy_->deviceMem(),
                                 loss_ ? (float *)loss_->deviceMem() : NULL,
                                 c, scale * p.mp_scaling_,
                                 p.ctx_->stream_);

      break;
    default:
      return "Unsupported tensor datatype";
    }
    return NULL;
  }

  std::vector<std::shared_ptr<CudaTensor>> getInputs() const override {
    return {x_, y_, dy_};
  }

  std::vector<std::shared_ptr<CudaTensor>> getOutputs() const override {
    return {dx_, loss_};
  }

};

static const char *
catclassifier_setup(CudaProgram &p, const Node &n, bool training)
{
  auto x = p.lower_tensor_batch(n.inputs_.get("x"));
  auto y = p.lower_tensor_batch(n.outputs_.get("y"));

  auto op = std::make_shared<CudaCatClassifierFwd>(x, y);

  if(!training) {
    p.infer(op);
    return NULL;
  }
  p.fwd(op);

  auto dx = x->grad_;
  if(!dx)
    return NULL;
  auto dy = y->makeSharedGrad();
  auto loss = p.lower_tensor_batch(n.outputs_.get("loss"));

  p.bwd(std::make_shared<CudaCatClassifierBwd>(x, y, dx, dy, loss));
  return NULL;
}


REGISTER_CUDA_OP("catclassifier", catclassifier_setup);





//------------------------------------------------------------------------


struct CudnnDropoutFwd : public CudnnOperation {
  const std::shared_ptr<CudaContext> ctx_;
  const std::shared_ptr<CudaTensor> x_, y_;
  cudnnDropoutDescriptor_t desc_;
  size_t reserve_size_;
  void *reserve_;
  size_t states_size_;
  void *states_;

  CudnnDropoutFwd(CudaProgram &p,
                  std::shared_ptr<CudaTensor> x,
                  std::shared_ptr<CudaTensor> y,
                  float prob)
    : CudnnOperation("dropoutfwd")
    , ctx_(p.ctx_), x_(x), y_(y)
  {
    chkCUDNN(cudnnDropoutGetReserveSpaceSize(x_->desc(), &reserve_size_));
    chkCuda(cudaMalloc(&reserve_, reserve_size_));

    chkCUDNN(cudnnDropoutGetStatesSize(p.ctx_->cudnn_, &states_size_));
    chkCuda(cudaMalloc(&states_, states_size_));

    chkCUDNN(cudnnCreateDropoutDescriptor(&desc_));
    chkCUDNN(cudnnSetDropoutDescriptor(desc_, p.ctx_->cudnn_, prob,
                                       states_, states_size_, 0));
  }

  ~CudnnDropoutFwd()
  {
    chkCUDNN(cudnnDestroyDropoutDescriptor(desc_));

    chkCuda(cudaFree(states_));
    chkCuda(cudaFree(reserve_));
  }

  cudnnStatus_t exec(CudaProgram &p) {
    return cudnnDropoutForward(p.ctx_->cudnn_, desc_,
                               x_->desc(), x_->deviceMem(),
                               y_->desc(), y_->deviceMem(),
                               reserve_, reserve_size_);
  }

  std::vector<std::shared_ptr<CudaTensor>> getInputs() const override {
    return {x_};
  }

  std::vector<std::shared_ptr<CudaTensor>> getOutputs() const override {
    return {y_};
  }

};


struct CudnnDropoutBwd : public CudnnOperation {
  const std::shared_ptr<CudnnDropoutFwd> fwd_;
  const std::shared_ptr<CudaTensor> dx_, dy_;

  CudnnDropoutBwd(std::shared_ptr<CudnnDropoutFwd> fwd,
                  std::shared_ptr<CudaTensor> dx,
                  std::shared_ptr<CudaTensor> dy)
    : CudnnOperation("dropoutbwd")
    , fwd_(fwd), dx_(dx), dy_(dy)
  {}

  ~CudnnDropoutBwd()
  {
  }

  cudnnStatus_t exec(CudaProgram &p) {
    return cudnnDropoutBackward(p.ctx_->cudnn_, fwd_->desc_,
                                dy_->desc(), dy_->deviceMem(),
                                dx_->desc(), dx_->deviceMem(),
                                fwd_->reserve_, fwd_->reserve_size_);
  }

  std::vector<std::shared_ptr<CudaTensor>> getInputs() const override {
    return {dy_};
  }

  std::vector<std::shared_ptr<CudaTensor>> getOutputs() const override {
    return {dx_};
  }
};



static const char *
dropout_setup(CudaProgram &p, const Node &n, bool training)
{
  assert(training);

  auto x = p.lower_tensor_batch(n.inputs_.get("x"));
  auto y = p.lower_tensor_batch(n.outputs_.get("y"));
  const float prob = n.attributes_.get("prob", 0.5f);

  auto fwd = std::make_shared<CudnnDropoutFwd>(p, x, y, prob);
  p.fwd(fwd);

  auto dx = x->grad_;
  if(!dx)
    return NULL;
  auto dy = y->makeSharedGrad();

  const float dx_beta = n.attributes_.get("dx.beta", 0.0f);
  if(dx_beta)
    return "dropout backward with dx_beta != 0";

  p.bwd(std::make_shared<CudnnDropoutBwd>(fwd, dx, dy));
  return NULL;
}



static std::vector<std::shared_ptr<Node>>
dropout_transform_node(CudaProgram &p, std::shared_ptr<Node> n)
{
  auto y = n->outputs_.get("y");
  auto ly = p.tensors_[y];

  if(ly) {
    auto x = n->inputs_.get("x");
    auto lx = std::make_shared<CudaTensor>(ly->storage_,
                                           ly->dims_, p.tensorFormat(ly->data_type_),
                                           ly->namePostfix("dropout"));
    p.tensors_[x] = ly;

  } else {

    auto x = p.lower_tensor_batch(n->inputs_.get("x"));
    ly = std::make_shared<CudaTensor>(x->storage_,
                                      x->dims_, p.tensorFormat(x->data_type_),
                                      x->namePostfix("dropout"));
    p.tensors_[y] = ly;
  }
  return {};
}




REGISTER_CUDA_OP("dropout", dropout_setup);




static std::vector<std::shared_ptr<Node>>
dropout_transform(CudaProgram &p,
                  const std::vector<std::shared_ptr<Node>> &nodes)
{
  std::vector<std::shared_ptr<Node>> r;

  for(size_t i = 0; i < nodes.size(); i++) {
    auto &n = nodes[i];
    if(n->type_ == "dropout") {
      dropout_transform_node(p, n);
    } else {
      r.push_back(n);
    }
  }
  return r;
}

REGISTER_CUDA_TRANSFORM(500, CUDA_TRANSFORM_INFERENCE, dropout_transform);








//------------------------------------------------------------------------

struct CudnnBatchNormInference : public CudnnOperation {

  const std::shared_ptr<CudaTensor> x_, s_, b_, m_, v_, y_;
  const float epsilon_;

  CudnnBatchNormInference(std::shared_ptr<CudaTensor> x,
                          std::shared_ptr<CudaTensor> s,
                          std::shared_ptr<CudaTensor> b,
                          std::shared_ptr<CudaTensor> m,
                          std::shared_ptr<CudaTensor> v,
                          std::shared_ptr<CudaTensor> y,
                          float epsilon)
    : CudnnOperation("bninf")
    , x_(x), s_(s), b_(b), m_(m), v_(v), y_(y), epsilon_(epsilon)
  {}

  cudnnStatus_t exec(CudaProgram &p) {
    float alpha = 1.0f;
    float beta = 0.0f;

    return cudnnBatchNormalizationForwardInference(p. ctx_->cudnn_,
                                                   CUDNN_BATCHNORM_SPATIAL,
                                                   &alpha, &beta,
                                                   x_->desc(),
                                                   x_->deviceMem(),
                                                   y_->desc(),
                                                   y_->deviceMem(),
                                                   s_->desc(),
                                                   s_->deviceMem(),
                                                   b_->deviceMem(),
                                                   m_->deviceMem(),
                                                   v_->deviceMem(),
                                                   epsilon_);
  }


  std::vector<std::shared_ptr<CudaTensor>> getInputs() const override {
    return {x_, s_, b_, m_, v_};
  }

  std::vector<std::shared_ptr<CudaTensor>> getOutputs() const override {
    return {y_};
  }

};


struct CudnnBatchNormTrain : public CudnnOperation {

  const std::shared_ptr<CudaTensor> x_, s_, b_, m_, v_, y_, sm_, sv_;
  const float epsilon_;
  const float expavgf_;

  CudnnBatchNormTrain(std::shared_ptr<CudaTensor> x,
                      std::shared_ptr<CudaTensor> s,
                      std::shared_ptr<CudaTensor> b,
                      std::shared_ptr<CudaTensor> m,
                      std::shared_ptr<CudaTensor> v,
                      std::shared_ptr<CudaTensor> y,
                      std::shared_ptr<CudaTensor> sm,
                      std::shared_ptr<CudaTensor> sv,
                      float epsilon,
                      float expavgf)
    : CudnnOperation("bntrain")
    , x_(x), s_(s), b_(b), m_(m), v_(v), y_(y), sm_(sm), sv_(sv)
    , epsilon_(epsilon), expavgf_(expavgf)
  {}

  cudnnStatus_t exec(CudaProgram &p) {

    float alpha = 1.0f;
    float beta = 0.0f;

    return cudnnBatchNormalizationForwardTraining(p.ctx_->cudnn_,
                                                  CUDNN_BATCHNORM_SPATIAL,
                                                  &alpha, &beta,
                                                  x_->desc(),
                                                  x_->deviceMem(),
                                                  y_->desc(),
                                                  y_->deviceMem(),
                                                  s_->desc(),
                                                  s_->deviceMem(),
                                                  b_->deviceMem(),
                                                  expavgf_,
                                                  m_->deviceMem(),
                                                  v_->deviceMem(),
                                                  epsilon_,
                                                  sm_->deviceMem(),
                                                  sv_->deviceMem());
  }


  std::vector<std::shared_ptr<CudaTensor>> getInputs() const override {
    return {x_, s_, b_, m_, v_};
  }

  std::vector<std::shared_ptr<CudaTensor>> getOutputs() const override {
    return {y_, sm_, sv_};
  }

};


struct CudnnBatchNormBwd : public CudnnOperation {

  const std::shared_ptr<CudaTensor> x_, dy_, dx_, s_, ds_, db_, sm_, sv_;
  const float epsilon_;
  const float dx_beta_;

  CudnnBatchNormBwd(CudnnBatchNormTrain &fwd,
                    std::shared_ptr<CudaTensor> dy,
                    std::shared_ptr<CudaTensor> dx,
                    std::shared_ptr<CudaTensor> ds,
                    std::shared_ptr<CudaTensor> db,
                    float dx_beta)
    : CudnnOperation("bnbwd")
    , x_(fwd.x_), dy_(dy), dx_(dx), s_(fwd.s_), ds_(ds), db_(db)
    , sm_(fwd.sm_), sv_(fwd.sv_), epsilon_(fwd.epsilon_)
    , dx_beta_(dx_beta)
  {}

  cudnnStatus_t exec(CudaProgram &p) {
    float alpha = 1.0f, beta = 0.0f;

    return cudnnBatchNormalizationBackward(p.ctx_->cudnn_,
                                           CUDNN_BATCHNORM_SPATIAL,
                                           &alpha, &dx_beta_,
                                           &alpha, &beta,
                                           x_->desc(),
                                           x_->deviceMem(),
                                           dy_->desc(),
                                           dy_->deviceMem(),
                                           dx_->desc(),
                                           dx_->deviceMem(),
                                           s_->desc(),
                                           s_->deviceMem(),
                                           ds_->deviceMem(),
                                           db_->deviceMem(),
                                           epsilon_,
                                           sm_->deviceMem(),
                                           sv_->deviceMem());
  }

  std::vector<std::shared_ptr<CudaTensor>> getInputs() const override {
    auto r = std::vector{x_, dy_, s_, sm_, sv_};
    if(dx_beta_)
      r.push_back(dx_);
    return r;
  }

  std::vector<std::shared_ptr<CudaTensor>> getOutputs() const override {
    return {dx_, ds_, db_};
  }
};


static const char *
batchnorm_setup(CudaProgram &p, const Node &n, bool training)
{
  auto x = p.lower_tensor_batch(n.inputs_.get("x"));
  auto y = p.lower_tensor_batch(n.outputs_.get("y"));

  auto s = p.lower_tensor(n.inputs_.get("s"), 2);
  auto b = p.lower_tensor(n.inputs_.get("b"), 2);
  auto m = p.lower_tensor(n.inputs_.get("m"), 2);
  auto v = p.lower_tensor(n.inputs_.get("v"), 2);
  const float epsilon = n.attributes_.get("epsilon", 1e-5f);

  if(!training) {
    p.infer(std::make_shared<CudnnBatchNormInference>(x, s, b, m, v,
                                                      y, epsilon));
    return NULL;
  }

  auto sm = std::make_shared<CudaTensor>(*m, m->namePostfix("smean"));
  auto sv = std::make_shared<CudaTensor>(*v, v->namePostfix("svar"));

  const float expavgf = n.attributes_.get("expavgf", 0.1f);
  const float dx_beta = n.attributes_.get("dx.beta", 0.0f);
  auto f = std::make_shared<CudnnBatchNormTrain>(x, s, b, m, v, y,
                                                 sm, sv, epsilon, expavgf);
  p.fwd(f);

  auto dx = x->makeSharedGrad();
  auto dy = y->makeSharedGrad();
  auto ds = s->makePrivateGrad();
  auto db = b->makePrivateGrad();

  p.bwd(std::make_shared<CudnnBatchNormBwd>(*f, dy, dx, ds, db, dx_beta));

  p.upd(std::make_shared<CudnnAdam>(p, s, ds));
  p.upd(std::make_shared<CudnnAdam>(p, b, db));
  return NULL;
}



REGISTER_CUDA_OP("batchnorm", batchnorm_setup);


//------------------------------------------------------------------------

struct CudnnOpTensor : public CudnnOperation {

  const std::shared_ptr<CudaTensor> a_, b_, c_;
  cudnnOpTensorDescriptor_t desc_;

  const char *opname(cudnnOpTensorOp_t op)
  {
    switch(op) {
    case CUDNN_OP_TENSOR_ADD:  return "add";
    case CUDNN_OP_TENSOR_MUL:  return "mul";
    case CUDNN_OP_TENSOR_MIN:  return "min";
    case CUDNN_OP_TENSOR_MAX:  return "max";
    case CUDNN_OP_TENSOR_SQRT: return "sqrt";
    case CUDNN_OP_TENSOR_NOT:  return "not";
    default:
      abort();
    }
  }

  CudnnOpTensor(std::shared_ptr<CudaTensor> a,
                std::shared_ptr<CudaTensor> b,
                std::shared_ptr<CudaTensor> c,
                cudnnOpTensorOp_t op)
    : CudnnOperation(opname(op))
    , a_(a), b_(b), c_(c)
  {
    cudnnCreateOpTensorDescriptor(&desc_);
    cudnnSetOpTensorDescriptor(desc_, op, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN);
  }

  ~CudnnOpTensor()
  {
    cudnnDestroyOpTensorDescriptor(desc_);
  }

  cudnnStatus_t exec(CudaProgram &p) {

    float alpha = 1.0f;
    float beta = 0.0f;

    return cudnnOpTensor(p.ctx_->cudnn_, desc_,
                         &alpha, a_->desc(), a_->deviceMem(),
                         &alpha, b_->desc(), b_->deviceMem(),
                         &beta,  c_->desc(), c_->deviceMem());
  }

  std::vector<std::shared_ptr<CudaTensor>> getInputs() const override {
    return {a_, b_};
  }

  std::vector<std::shared_ptr<CudaTensor>> getOutputs() const override {
    return {c_};
  }
};

static const char *
sum_setup(CudaProgram &p, const Node &n, bool training)
{
  if(training)
    return "not supported for training";

  auto x0 = p.lower_tensor_batch(n.inputs_.get("x0"));
  auto x1 = p.lower_tensor_batch(n.inputs_.get("x1"));

  auto y = p.lower_tensor_batch(n.outputs_.get("y"));

  p.infer(std::make_shared<CudnnOpTensor>(x0, x1, y, CUDNN_OP_TENSOR_ADD));
  return NULL;
}

REGISTER_CUDA_OP("sum", sum_setup);


//------------------------------------------------------------------------

struct CudnnSoftmaxFwd : public CudnnOperation {

  const std::shared_ptr<CudaTensor> x_, y_;

  CudnnSoftmaxFwd(std::shared_ptr<CudaTensor> x,
                  std::shared_ptr<CudaTensor> y)
    : CudnnOperation("softmaxfwd")
    , x_(x), y_(y)
  {}

  cudnnStatus_t exec(CudaProgram &p) {

    float alpha = 1.0f, beta = 0.0f;

    return cudnnSoftmaxForward(p.ctx_->cudnn_,
                               CUDNN_SOFTMAX_ACCURATE,
                               CUDNN_SOFTMAX_MODE_CHANNEL,
                               &alpha,
                               x_->desc(), x_->deviceMem(),
                               &beta,
                               y_->desc(), y_->deviceMem());
  }

  std::vector<std::shared_ptr<CudaTensor>> getInputs() const override {
    return {x_};
  }

  std::vector<std::shared_ptr<CudaTensor>> getOutputs() const override {
    return {y_};
  }

};

static const char *
softmax_setup(CudaProgram &p, const Node &n, bool training)
{
  if(training)
    return "not supported for training";

  auto x = p.lower_tensor_batch(n.inputs_.get("x"));
  auto y = p.lower_tensor_batch(n.outputs_.get("y"));
  p.infer(std::make_shared<CudnnSoftmaxFwd>(x, y));
  return NULL;
}

REGISTER_CUDA_OP("softmax", softmax_setup);



//------------------------------------------------------------------------

struct CudnnBatchNormActTrain : public CudnnOperation {

  const std::shared_ptr<CudaContext> ctx_;
  const std::shared_ptr<CudaTensor> x_, s_, b_, m_, v_, y_, sm_, sv_;
  const float epsilon_;
  const float expavgf_;
  const cudnnBatchNormOps_t ops_;
  const cudnnBatchNormMode_t mode_;
  cudnnActivationDescriptor_t desc_;

  void *reserve_;
  size_t reserve_size_;

  CudnnBatchNormActTrain(CudaProgram &p,
                         std::shared_ptr<CudaTensor> x,
                         std::shared_ptr<CudaTensor> s,
                         std::shared_ptr<CudaTensor> b,
                         std::shared_ptr<CudaTensor> m,
                         std::shared_ptr<CudaTensor> v,
                         std::shared_ptr<CudaTensor> y,
                         std::shared_ptr<CudaTensor> sm,
                         std::shared_ptr<CudaTensor> sv,
                         float epsilon,
                         float expavgf,
                         cudnnBatchNormOps_t ops,
                         cudnnActivationMode_t activation_mode,
                         float actalpha)
    : CudnnOperation("bnacttrain")
    , ctx_(p.ctx_)
    , x_(x), s_(s), b_(b), m_(m), v_(v), y_(y), sm_(sm), sv_(sv)
    , epsilon_(epsilon), expavgf_(expavgf), ops_(ops)
    , mode_(CUDNN_BATCHNORM_SPATIAL_PERSISTENT)
  {
    chkCUDNN(cudnnCreateActivationDescriptor(&desc_));
    chkCUDNN(cudnnSetActivationDescriptor(desc_, activation_mode,
                                          CUDNN_PROPAGATE_NAN, actalpha));

    chkCUDNN(cudnnGetBatchNormalizationTrainingExReserveSpaceSize(p.ctx_->cudnn_,
                                                                  mode_, ops_,
                                                                  desc_,
                                                                  x_->desc(),
                                                                  &reserve_size_));
    chkCuda(cudaMalloc(&reserve_, reserve_size_));
  }

  ~CudnnBatchNormActTrain()
  {
    chkCUDNN(cudnnDestroyActivationDescriptor(desc_));
    chkCuda(cudaFree(reserve_));
  }

  cudnnStatus_t exec(CudaProgram &p) {

    float alpha = 1.0f;
    float beta = 0.0f;

    return cudnnBatchNormalizationForwardTrainingEx(ctx_->cudnn_,
                                                    mode_, ops_,
                                                    &alpha, &beta,
                                                    x_->desc(),
                                                    x_->deviceMem(),
                                                    NULL, NULL,
                                                    y_->desc(),
                                                    y_->deviceMem(),
                                                    s_->desc(),
                                                    s_->deviceMem(),
                                                    b_->deviceMem(),
                                                    expavgf_,
                                                    m_->deviceMem(),
                                                    v_->deviceMem(),
                                                    epsilon_,
                                                    sm_->deviceMem(),
                                                    sv_->deviceMem(),
                                                    desc_,
                                                    p.workspace_.ptr(),
                                                    p.workspace_.size(),
                                                    reserve_, reserve_size_);
  }


  std::vector<std::shared_ptr<CudaTensor>> getInputs() const override {
    return {x_, s_, b_, m_, v_};
  }

  std::vector<std::shared_ptr<CudaTensor>> getOutputs() const override {
    return {y_, sm_, sv_};
  }

};


struct CudnnBatchNormActBwd : public CudnnOperation {

  const std::shared_ptr<CudnnBatchNormActTrain> fwd_;
  const std::shared_ptr<CudaTensor> dy_, dx_, ds_, db_;
  const float dx_beta_;

  CudnnBatchNormActBwd(CudaProgram &p,
                       std::shared_ptr<CudnnBatchNormActTrain> fwd,
                       std::shared_ptr<CudaTensor> dy,
                       std::shared_ptr<CudaTensor> dx,
                       std::shared_ptr<CudaTensor> ds,
                       std::shared_ptr<CudaTensor> db,
                       float dx_beta)
    : CudnnOperation("bnactbwd")
    , fwd_(fwd)
    , dy_(dy), dx_(dx), ds_(ds), db_(db)
    , dx_beta_(dx_beta)
  {
    size_t workspace;
    chkCUDNN(cudnnGetBatchNormalizationBackwardExWorkspaceSize(p.ctx_->cudnn_,
                                                               fwd->mode_,
                                                               fwd->ops_,
                                                               fwd->x_->desc(),
                                                               fwd->y_->desc(),
                                                               dy_->desc(),
                                                               NULL,
                                                               dx_->desc(),
                                                               ds_->desc(),
                                                               fwd->desc_,
                                                               &workspace));
    p.workspace_.request(workspace);


  }

  cudnnStatus_t exec(CudaProgram &p) {
    float alpha = 1.0f, beta = 0.0f;

    return cudnnBatchNormalizationBackwardEx(p.ctx_->cudnn_,
                                             fwd_->mode_, fwd_->ops_,
                                             &alpha, &dx_beta_,
                                             &alpha, &beta,
                                             fwd_->x_->desc(),
                                             fwd_->x_->deviceMem(),
                                             fwd_->y_->desc(),
                                             fwd_->y_->deviceMem(),
                                             dy_->desc(),
                                             dy_->deviceMem(),
                                             NULL, NULL,
                                             dx_->desc(),
                                             dx_->deviceMem(),
                                             fwd_->s_->desc(),
                                             fwd_->s_->deviceMem(),
                                             fwd_->b_->deviceMem(),
                                             ds_->deviceMem(),
                                             db_->deviceMem(),
                                             fwd_->epsilon_,
                                             fwd_->sm_->deviceMem(),
                                             fwd_->sv_->deviceMem(),
                                             fwd_->desc_,
                                             p.workspace_.ptr(),
                                             p.workspace_.size(),
                                             fwd_->reserve_,
                                             fwd_->reserve_size_);

  }

  std::vector<std::shared_ptr<CudaTensor>> getInputs() const override {
    auto r = std::vector{fwd_->x_, fwd_->y_, dy_, fwd_->s_,
                         fwd_->b_, fwd_->sm_, fwd_->sv_};
    if(dx_beta_)
      r.push_back(dx_);
    return r;
  }

  std::vector<std::shared_ptr<CudaTensor>> getOutputs() const override {
    return {dx_, ds_, db_};
  }
};






static const char *
batchnorm_relu_setup(CudaProgram &p, const Node &n, bool training)
{
  if(!training)
    return "not supported for inferenece";

  auto x = p.lower_tensor_batch(n.inputs_.get("x"));
  auto y = p.lower_tensor_batch(n.outputs_.get("y"));

  auto s = p.lower_tensor(n.inputs_.get("s"), 2);
  auto b = p.lower_tensor(n.inputs_.get("b"), 2);
  auto m = p.lower_tensor(n.inputs_.get("m"), 2);
  auto v = p.lower_tensor(n.inputs_.get("v"), 2);
  const float epsilon = n.attributes_.get("epsilon", 1e-5f);

  auto sm = std::make_shared<CudaTensor>(*m, m->namePostfix("smean"));
  auto sv = std::make_shared<CudaTensor>(*v, v->namePostfix("svar"));

  const float expavgf = n.attributes_.get("expavgf", 0.1f);
  const float dx_beta = n.attributes_.get("dx.beta", 0.0f);

  auto ops = CUDNN_BATCHNORM_OPS_BN_ACTIVATION;
  auto activation_mode = CUDNN_ACTIVATION_RELU;
  float activation_alpha = 0.0f;

  auto f = std::make_shared<CudnnBatchNormActTrain>(p, x, s, b, m, v, y,
                                                    sm, sv, epsilon, expavgf,
                                                    ops, activation_mode,
                                                    activation_alpha);
  p.fwd(f);

  auto dx = x->makeSharedGrad();
  auto dy = y->makeSharedGrad();
  auto ds = s->makePrivateGrad();
  auto db = b->makePrivateGrad();

  p.bwd(std::make_shared<CudnnBatchNormActBwd>(p, f, dy, dx, ds, db, dx_beta));

  p.upd(std::make_shared<CudnnAdam>(p, s, ds));
  p.upd(std::make_shared<CudnnAdam>(p, b, db));
  return NULL;
}

REGISTER_CUDA_OP("batchnorm_relu", batchnorm_relu_setup);





static std::shared_ptr<Node>
batchnorm_relu_transform_node(CudaProgram &p,
                              std::shared_ptr<Node> bn,
                              std::shared_ptr<Node> mp)
{
  auto x = bn->inputs_["x"];
  auto y = mp->outputs_["y"];

  if(x->data_type_ != Tensor::DataType::HALF)
    return nullptr;

  auto lx = p.tensors_.find(x);
  if(lx != p.tensors_.end()) {
    if(!lx->second->cpacked())
      return nullptr;
  }

  auto ly = p.tensors_.find(y);
  if(ly != p.tensors_.end()) {
    if(!ly->second->cpacked())
      return nullptr;
  }


  auto nn = std::make_shared<Node>("batchnorm_relu");

  nn->inputs_["x"] = bn->inputs_["x"];
  nn->inputs_["s"] = bn->inputs_["s"];
  nn->inputs_["b"] = bn->inputs_["b"];
  nn->inputs_["m"] = bn->inputs_["m"];
  nn->inputs_["v"] = bn->inputs_["v"];

  if(bn->attributes_.find("epsilon") != bn->attributes_.end())
    nn->attributes_["epsilon"] = bn->attributes_["epsilon"];

  nn->outputs_["y"] = mp->outputs_["y"];
  return nn;
}





static std::vector<std::shared_ptr<Node>>
batchnorm_relu_transform(CudaProgram &p,
                         const std::vector<std::shared_ptr<Node>> &nodes)
{
  std::vector<std::shared_ptr<Node>> r;

  if(nodes.size() < 2)
    return nodes;

  for(size_t i = 0; i < nodes.size(); i++) {
    std::shared_ptr<Node> n = nodes[i];

    if(i < nodes.size() - 1 &&
       nodes[i + 0]->type_ == "batchnorm" &&
       nodes[i + 1]->type_ == "relu") {
      auto n2 = batchnorm_relu_transform_node(p, nodes[i], nodes[i + 1]);
      if(n2) {
        i++;
        n = n2;
      }
    }
    r.push_back(n);
  }
  return r;
}

REGISTER_CUDA_TRANSFORM(500, CUDA_TRANSFORM_TRAINING, batchnorm_relu_transform);


//------------------------------------------------------------------------

struct CudnnSpatialTransformFwd : public CudnnOperation {

  const std::shared_ptr<CudaTensor> x_, theta_, y_, grid_;
  cudnnSpatialTransformerDescriptor_t desc_;

  CudnnSpatialTransformFwd(std::shared_ptr<CudaContext> ctx,
                           std::shared_ptr<CudaTensor> x,
                           std::shared_ptr<CudaTensor> theta,
                           std::shared_ptr<CudaTensor> y)
    : CudnnOperation("spatialtransform")
    , x_(x), theta_(theta), y_(y)
    , grid_(std::make_shared<CudaTensor>(Tensor::DataType::FLOAT,
                                         Dims{y_->dims_[0],
                                             2,
                                             y_->dims_[2],
                                             y_->dims_[3]},
                                         CUDNN_TENSOR_NHWC,
                                         ctx))
  {
    int dims[4] = {
      (int)y_->dims_[0], // n
      1, // c
      (int)y_->dims_[2], // h
      (int)y_->dims_[3]  // w
    };
    chkCUDNN(cudnnCreateSpatialTransformerDescriptor(&desc_));
    chkCUDNN(cudnnSetSpatialTransformerNdDescriptor(desc_,
                                                    CUDNN_SAMPLER_BILINEAR,
                                                    CUDNN_DATA_FLOAT,
                                                    4,
                                                    dims));
  }

  ~CudnnSpatialTransformFwd()
  {
    chkCUDNN(cudnnDestroySpatialTransformerDescriptor(desc_));
  }

  cudnnStatus_t exec(CudaProgram &p) {
    float alpha = 1.0f, beta = 0.0f;
    cudnnStatus_t s;

    s = cudnnSpatialTfGridGeneratorForward(p.ctx_->cudnn_, desc_,
                                           theta_->deviceMem(),
                                           grid_->deviceMem());
    if(s)
      return s;
    return cudnnSpatialTfSamplerForward(p.ctx_->cudnn_, desc_,
                                        &alpha,
                                        x_->desc(), x_->deviceMem(),
                                        grid_->deviceMem(),
                                        &beta,
                                        y_->desc(), y_->deviceMem());
  }

  std::vector<std::shared_ptr<CudaTensor>> getInputs() const override {
    return {x_, theta_};
  }

  std::vector<std::shared_ptr<CudaTensor>> getOutputs() const override {
    return {y_};
  }
};



static const char *
spatialtransform_setup(CudaProgram &p, const Node &n, bool training)
{
  auto x = p.lower_tensor_batch(n.inputs_.get("x"));

  bool bypass = !training && !n.attributes_.get("inference", false);

  if(bypass) {
    auto y = p.lower_tensor_batch(n.outputs_.get("y"));
    p.infer(std::make_shared<CudnnTransform>(x, y, 0.0f));
    return NULL;
  }

  auto y = p.lower_tensor_batch(n.outputs_.get("y"));
  auto theta = p.lower_tensor(n.inputs_.get("theta"));
  auto op = std::make_shared<CudnnSpatialTransformFwd>(p.ctx_, x, theta, y);

  if(training) {
    p.fwd(op);
  } else {
    p.infer(op);
  }
  return NULL;
}

REGISTER_CUDA_OP("spatialtransform", spatialtransform_setup);

//------------------------------------------------------------------------


static std::vector<std::shared_ptr<Node>>
reshape_transform_node(CudaProgram &p, std::shared_ptr<Node> n)
{
  auto x = p.lower_tensor_batch(n->inputs_.get("x"), CUDNN_TENSOR_NCHW);
  auto dx = x->makeSharedGrad();
  auto y = n->outputs_.get("y");

  auto yl = std::make_shared<CudaTensor>(x->storage_,
                                         y->dims_.n(p.batch_size_),
                                         CUDNN_TENSOR_NCHW,
                                         x->namePostfix("reshape"));

  p.tensors_[y] = yl;
  yl->grad_ = std::make_shared<CudaTensor>(dx->storage_,
                                           y->dims_.n(p.batch_size_),
                                           CUDNN_TENSOR_NCHW,
                                           x->namePostfix("reshape"));
  return {};
}



static std::vector<std::shared_ptr<Node>>
reshape_transform(CudaProgram &p,
                  const std::vector<std::shared_ptr<Node>> &nodes)
{
  std::vector<std::shared_ptr<Node>> r;

  for(size_t i = 0; i < nodes.size(); i++) {
    auto &n = nodes[i];
    if(n->type_ == "reshape") {
      reshape_transform_node(p, n);
    } else {
      r.push_back(n);
    }
  }
  return r;
}

REGISTER_CUDA_TRANSFORM(110, CUDA_TRANSFORM_ALL, reshape_transform);



//------------------------------------------------------------------------

static void
concat_transform_node(CudaProgram &p, const Node &n)
{
  const int axis = 1;

  auto y = p.lower_tensor_batch(n.outputs_.get("y"));
  auto dy = y->makeSharedGrad();
  auto element_offset = std::vector<int64_t>(y->dims_.size(), 0);

  for(const auto &xh : n.inputs_.getv("x")) {
    auto x = std::make_shared<CudaTensor>(y, xh->dims_.n(p.batch_size_),
                                          element_offset,
                                          xh->namePostfix("alias"));
    x->copyFromLocked(*xh);
    p.tensors_[xh] = x;
    x->grad_ = std::make_shared<CudaTensor>(dy, xh->dims_.n(p.batch_size_),
                                            element_offset,
                                            xh->namePostfix("alias"));
    element_offset[axis] += xh->dims_[axis];
  }
}



static std::vector<std::shared_ptr<Node>>
concat_transform(CudaProgram &p,
                 const std::vector<std::shared_ptr<Node>> &nodes)
{
  std::vector<std::shared_ptr<Node>> r;

  for(ssize_t i = nodes.size() - 1; i >= 0; i--) {
    auto &n = nodes[i];
    if(n->type_ == "concat") {
      concat_transform_node(p, *n);
    } else {
      r.insert(r.begin(), n);
    }
  }
  return r;
}

REGISTER_CUDA_TRANSFORM(100, CUDA_TRANSFORM_ALL, concat_transform);


}

