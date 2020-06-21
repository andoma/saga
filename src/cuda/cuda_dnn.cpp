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


//------------------------------------------------------------------------
struct CudnnAdam : public CudaOperation {

  const std::shared_ptr<CudaTensor> weights_, gradient_;
  float learning_rate_;
  float *temp_;
  int iter_;

  CudnnAdam(CudaProgram &p,
            std::shared_ptr<CudaTensor> weights,
            std::shared_ptr<CudaTensor> gradient)
    : weights_(weights)
    , gradient_(gradient)
    , learning_rate_(p.learning_rate_)
    , iter_(0)
  {
    assert(weights->dims_ == gradient->dims_);

    size_t bytes;
    switch(weights->data_type_) {
    case Tensor::DataType::FLOAT:
      // Allocate 2x floats for each weight (m and v)
      bytes = weights_->elements_ * 2 * sizeof(float);
      chkCuda(cudaMalloc(&temp_, bytes));
      chkCuda(cudaMemset(temp_, 0, bytes));
      break;

    case Tensor::DataType::HALF:
      // Allocate 3x floats for each weight (m and v and float32 copy)
      bytes = weights_->elements_ * 3 * sizeof(float);
      chkCuda(cudaMallocManaged(&temp_, bytes, cudaMemAttachGlobal));
      {
        const uint16_t *src = (const uint16_t *)weights->deviceMem();
        float *dst = temp_;
        for(int i = 0; i < weights->elements_; i++) {
          *dst++ = 0;
          *dst++ = 0;
          *dst++ = _cvtsh_ss(*src++);
        }
        break;
      }


    default:
      abort();
    }
  }

  ~CudnnAdam()
  {
    chkCuda(cudaFree(temp_));
  }

  void print() const {
    printf("Adam\n");
    printf("\tweights:  %s\n", weights_->info().c_str());
    printf("\tgradient: %s\n", gradient_->info().c_str());
  }

  void exec(CudaProgram &p) {
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
      abort();
    }
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



struct CudnnConvolutionFwd : public CudaOperation {

  const std::shared_ptr<CudaContext> ctx_;
  const std::shared_ptr<CudaTensor> x_, w_, b_, y_;
  const float y_beta_;

  cudnnConvolutionDescriptor_t conv_desc_;
  cudnnFilterDescriptor_t filter_desc_;
  cudnnConvolutionFwdAlgo_t conv_fwd_algo_;

  ~CudnnConvolutionFwd()
  {
    chkCUDNN(cudnnDestroyFilterDescriptor(filter_desc_));
    chkCUDNN(cudnnDestroyConvolutionDescriptor(conv_desc_));
  }

  CudnnConvolutionFwd(CudaProgram &p, const Node &n)
    : ctx_(p.ctx_)
    , x_(p.lower_tensor_batch(n.inputs_.get("x")))
    , w_(p.lower_tensor(n.inputs_.get("w")))
    , b_(p.lower_tensor(n.inputs_.get("b"), 2))
    , y_(p.lower_tensor_batch(n.outputs_.get("y")))
    , y_beta_(n.attributes_.get("y.beta", 0.0f))
  {
    chkCUDNN(cudnnCreateFilterDescriptor(&filter_desc_));
    chkCUDNN(cudnnCreateConvolutionDescriptor(&conv_desc_));


    chkCUDNN(cudnnSetFilter4dDescriptor(filter_desc_,
                                        x_->type_,
                                        p.tensorFormat(x_->data_type_),
                                        w_->dims_[0],
                                        w_->dims_[1],
                                        w_->dims_[2],
                                        w_->dims_[3]));

    chkCUDNN(cudnnSetConvolutionMathType(conv_desc_,
                                         CUDNN_TENSOR_OP_MATH));

    const int pad = n.attributes_.get("pad", 0);
    const int stride = n.attributes_.get("stride", 1);

    chkCUDNN(cudnnSetConvolution2dDescriptor(conv_desc_,
                                             pad, pad,
                                             stride, stride,
                                             1, 1,
                                             CUDNN_CROSS_CORRELATION,
                                             CUDNN_DATA_FLOAT));

    chkCUDNN(cudnnGetConvolutionForwardAlgorithm(ctx_->cudnn_,
                                                 x_->desc_,
                                                 filter_desc_,
                                                 conv_desc_,
                                                 y_->desc_,
                                                 CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                                 0,
                                                 &conv_fwd_algo_));

    size_t workspace;
    chkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(ctx_->cudnn_,
                                                     x_->desc_,
                                                     filter_desc_,
                                                     conv_desc_,
                                                     y_->desc_,
                                                     conv_fwd_algo_,
                                                     &workspace));

    p.requetstWorkspace(workspace);
  }


  void print() const {
    printf("Convolution Fwd %s\n", convfwdalgostr(conv_fwd_algo_));
    printf("\tx: %s\n", x_->info().c_str());
    printf("\tw: %s\n", w_->info().c_str());
    if(b_)
      printf("\tb: %s\n", b_->info().c_str());
    printf("\ty: %s\n", y_->info().c_str());
  }

  void exec(CudaProgram &p) {
    float alpha = 1.0f;

    chkCUDNN(cudnnConvolutionForward(ctx_->cudnn_, &alpha,
                                     x_->desc(),
                                     x_->deviceMem(),
                                     filter_desc_,
                                     w_->deviceMem(),
                                     conv_desc_,
                                     conv_fwd_algo_,
                                     p.workspace_, p.workspace_size_,
                                     &y_beta_,
                                     y_->desc(),
                                     y_->deviceMem()));

    if(b_) {
      chkCUDNN(cudnnAddTensor(ctx_->cudnn_,
                              &alpha, b_->desc(), b_->deviceMem(),
                              &alpha, y_->desc(), y_->deviceMem()));
    }

    if(p.debug_) {
      x_->printStats("conv.x");
      w_->printStats("conv.w");
      if(b_)
        b_->printStats("conv.b");
      y_->printStats("conv.y");

    }

  }
};

struct CudnnConvolutionBwd : public CudaOperation {

  const std::shared_ptr<CudaContext> ctx_;
  const std::shared_ptr<CudnnConvolutionFwd> fwd_;
  const std::shared_ptr<CudaTensor> dx_, dw_, db_, dy_;
  const float dx_beta_;

  cudnnConvolutionBwdDataAlgo_t bwd_data_algo_;
  cudnnConvolutionBwdFilterAlgo_t bwd_filter_algo_;

  CudnnConvolutionBwd(CudaProgram &p,
                      const Node &n,
                      std::shared_ptr<CudnnConvolutionFwd> fwd)
    : ctx_(p.ctx_)
    , fwd_(fwd)
    , dx_(fwd_->x_->grad_)
    , dw_(fwd_->w_->makeGrad())
    , db_(fwd_->b_ ? fwd_->b_->makeGrad() : nullptr)
    , dy_(fwd_->y_->makeGrad())
    , dx_beta_(n.attributes_.get("dx.beta", 0.0f))
  {

    chkCUDNN(cudnnGetConvolutionBackwardDataAlgorithm(ctx_->cudnn_,
                                                      fwd->filter_desc_,
                                                      fwd->y_->desc(),
                                                      fwd->conv_desc_,
                                                      fwd->x_->desc(),
                                                      CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
                                                      0,
                                                      &bwd_data_algo_));



    size_t workspace_bytes;

    chkCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(ctx_->cudnn_,
                                                          fwd->filter_desc_,
                                                          fwd->y_->desc(),
                                                          fwd->conv_desc_,
                                                          fwd->x_->desc(),
                                                          bwd_data_algo_,
                                                          &workspace_bytes));

    p.requetstWorkspace(workspace_bytes);

    chkCUDNN(cudnnGetConvolutionBackwardFilterAlgorithm(ctx_->cudnn_,
                                                        fwd->x_->desc(),
                                                        fwd->y_->desc(),
                                                        fwd->conv_desc_,
                                                        fwd->filter_desc_,
                                                        CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,
                                                        0,
                                                        &bwd_filter_algo_));

    chkCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(ctx_->cudnn_,
                                                            fwd->x_->desc(),
                                                            fwd->y_->desc(),
                                                            fwd->conv_desc_,
                                                            fwd->filter_desc_,
                                                            bwd_filter_algo_,
                                                            &workspace_bytes));

    p.requetstWorkspace(workspace_bytes);
  }

  ~CudnnConvolutionBwd()
  {

  }

  void print() const {
    printf("Convolution Bwd Filter:%s Data:%s dx.beta:%f\n",
           convbwdfilteralgostr(bwd_filter_algo_),
           convbwddataalgostr(bwd_data_algo_),
           dx_beta_);
    printf("\tdy: %s\n", dy_->info().c_str());
    if(db_)
      printf("\tdb: %s\n", db_->info().c_str());
    printf("\tdw: %s\n", dw_->info().c_str());
    if(dx_)
      printf("\tdx: %s\n", dx_->info().c_str());
  }

  void exec(CudaProgram &p) {

    float alpha = 1.0f, beta = 0.0f;

    if(db_) {
      chkCUDNN(cudnnConvolutionBackwardBias(ctx_->cudnn_, &alpha,
                                            dy_->desc(),
                                            dy_->deviceMem(),
                                            &beta,
                                            db_->desc(),
                                            db_->deviceMem()));
    }

    chkCUDNN(cudnnConvolutionBackwardFilter(ctx_->cudnn_, &alpha,
                                            fwd_->x_->desc(),
                                            fwd_->x_->deviceMem(),
                                            dy_->desc(),
                                            dy_->deviceMem(),
                                            fwd_->conv_desc_,
                                            bwd_filter_algo_,
                                            p.workspace_, p.workspace_size_,
                                            &beta,
                                            fwd_->filter_desc_,
                                            dw_->deviceMem()));

    if(dx_ != NULL) {
      chkCUDNN(cudnnConvolutionBackwardData(ctx_->cudnn_, &alpha,
                                            fwd_->filter_desc_,
                                            fwd_->w_->deviceMem(),
                                            dy_->desc(),
                                            dy_->deviceMem(),
                                            fwd_->conv_desc_,
                                            bwd_data_algo_,
                                            p.workspace_, p.workspace_size_,
                                            &dx_beta_,
                                            dx_->desc(),
                                            dx_->deviceMem()));
    }

    if(p.debug_) {
      dy_->printStats("conv.dy");
      dw_->printStats("conv.dw");
      db_->printStats("conv.db");
      if(dx_)
        dx_->printStats("conv.dx");
    }
  }
};


static void
conv_infer(CudaProgram &p, const Node &n)
{
  p.infer(std::make_shared<CudnnConvolutionFwd>(p, n));
}


static void
conv_train(CudaProgram &p, const Node &n)
{
  auto f = std::make_shared<CudnnConvolutionFwd>(p, n);
  p.train(f);
  auto b = std::make_shared<CudnnConvolutionBwd>(p, n, f);
  p.bwd(b);

  p.upd(std::make_shared<CudnnAdam>(p, f->w_, b->dw_));
  if(f->b_)
    p.upd(std::make_shared<CudnnAdam>(p, f->b_, b->db_));
}

REGISTER_CUDA_OP("conv", conv_infer, conv_train);



//------------------------------------------------------------------------

struct CudnnBatchNormInference : public CudaOperation {

  const std::shared_ptr<CudaContext> ctx_;
  const std::shared_ptr<CudaTensor> x_, s_, b_, m_, v_, y_;
  const float epsilon_;
  const float y_beta_;

  CudnnBatchNormInference(CudaProgram &p, const Node &n)
    : ctx_(p.ctx_)
    , x_(p.lower_tensor_batch(n.inputs_.get("x")))
    , s_(p.lower_tensor(n.inputs_.get("s"), 2))
    , b_(p.lower_tensor(n.inputs_.get("b"), 2))
    , m_(p.lower_tensor(n.inputs_.get("m"), 2))
    , v_(p.lower_tensor(n.inputs_.get("v"), 2))
    , y_(p.lower_tensor_batch(n.outputs_.get("y")))
    , epsilon_(n.attributes_.get("epsilon", 1e-5f))
    , y_beta_(n.attributes_.get("y.beta", 0.0f))
  {}

  void print() const {
    printf("BatchNorm Fwd\n");
    printf("\tx: %s\n", x_->info().c_str());
    printf("\ts: %s\n", s_->info().c_str());
    printf("\tb: %s\n", b_->info().c_str());
    printf("\tm: %s\n", m_->info().c_str());
    printf("\tv: %s\n", v_->info().c_str());
    printf("\ty: %s\n", y_->info().c_str());
  }

  void exec(CudaProgram &p) {
    float alpha = 1.0f;
    chkCUDNN(cudnnBatchNormalizationForwardInference(ctx_->cudnn_,
                                                     CUDNN_BATCHNORM_SPATIAL,
                                                     &alpha, &y_beta_,
                                                     x_->desc(),
                                                     x_->deviceMem(),
                                                     y_->desc(),
                                                     y_->deviceMem(),
                                                     s_->desc(),
                                                     s_->deviceMem(),
                                                     b_->deviceMem(),
                                                     m_->deviceMem(),
                                                     v_->deviceMem(),
                                                     epsilon_));

  }
};


struct CudnnBatchNormTrain : public CudnnBatchNormInference {

  const std::shared_ptr<CudaTensor> sm_, sv_;
  const float expavgf_;

  CudnnBatchNormTrain(CudaProgram &p, const Node &n)
    : CudnnBatchNormInference(p, n)
    , sm_(std::make_shared<CudaTensor>(*m_, p.tensorFormat(m_->data_type_)))
    , sv_(std::make_shared<CudaTensor>(*v_, p.tensorFormat(v_->data_type_)))
    , expavgf_(n.attributes_.get("expavg", 0.1f))
  {}

  void print() const {
    CudnnBatchNormInference::print();
    printf("\tsm: %s\n", sm_->info().c_str());
    printf("\tsv: %s\n", sv_->info().c_str());
  }

  void exec(CudaProgram &p) {
    float alpha = 1.0f;
    chkCUDNN(cudnnBatchNormalizationForwardTraining(ctx_->cudnn_,
                                                    CUDNN_BATCHNORM_SPATIAL,
                                                    &alpha, &y_beta_,
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
                                                    sv_->deviceMem()));


  }
};

struct CudnnBatchNormBwd : public CudaOperation {

  const std::shared_ptr<CudaContext> ctx_;
  const std::shared_ptr<CudaTensor> x_, dy_, dx_, s_, ds_, db_, sm_, sv_;
  const float epsilon_;
  const float dx_beta_;

  CudnnBatchNormBwd(CudaProgram &p, const Node &n,
                    const CudnnBatchNormTrain &fwd)
    : ctx_(p.ctx_)
    , x_(fwd.x_)
    , dy_(fwd.y_->makeGrad())
    , dx_(fwd.x_->makeGrad())
    , s_(fwd.s_)
    , ds_(fwd.s_->makeGrad())
    , db_(fwd.b_->makeGrad())
    , sm_(fwd.sm_)
    , sv_(fwd.sv_)
    , epsilon_(fwd.epsilon_)
    , dx_beta_(n.attributes_.get("dx.beta", 0.0f))
  {}

  void print() const {
    printf("BatchNorm Bwd\n");
  }

  void exec(CudaProgram &p) {
    float alpha = 1.0f, beta = 0.0f;

    chkCUDNN(cudnnBatchNormalizationBackward(ctx_->cudnn_,
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
                                             sv_->deviceMem()));
  }

};

static void
batchnorm_infer(CudaProgram &p, const Node &n)
{
  p.infer(std::make_shared<CudnnBatchNormInference>(p, n));
}

static void
batchnorm_train(CudaProgram &p, const Node &n)
{
  auto f = std::make_shared<CudnnBatchNormTrain>(p, n);
  p.train(f);

  auto b = std::make_shared<CudnnBatchNormBwd>(p, n, *f);
  p.bwd(b);

  p.upd(std::make_shared<CudnnAdam>(p, f->s_, b->ds_));
  p.upd(std::make_shared<CudnnAdam>(p, f->b_, b->db_));
}

REGISTER_CUDA_OP("batchnorm", batchnorm_infer, batchnorm_train);


//------------------------------------------------------------------------





struct CudnnBatchNormActivationTrain : public CudaOperation {

  const std::shared_ptr<CudaContext> ctx_;
  const std::shared_ptr<CudaTensor> x_, s_, b_, m_, v_, y_, sm_, sv_;
  const float epsilon_;
  const float expavgf_;
  const float y_beta_;

  cudnnBatchNormMode_t mode_;
  cudnnBatchNormOps_t bnOps_;

  cudnnActivationDescriptor_t desc_;

  size_t reserve_size_;
  void *reserve_;

  CudnnBatchNormActivationTrain(CudaProgram &p, const Node &n)
    : ctx_(p.ctx_)
    , x_(p.lower_tensor_batch(n.inputs_.get("x")))
    , s_(p.lower_tensor(n.inputs_.get("s"), 2))
    , b_(p.lower_tensor(n.inputs_.get("b"), 2))
    , m_(p.lower_tensor(n.inputs_.get("m"), 2))
    , v_(p.lower_tensor(n.inputs_.get("v"), 2))
    , y_(p.lower_tensor_batch(n.outputs_.get("y")))
    , sm_(std::make_shared<CudaTensor>(*m_, p.tensorFormat(m_->data_type_)))
    , sv_(std::make_shared<CudaTensor>(*v_, p.tensorFormat(v_->data_type_)))
    , epsilon_(n.attributes_.get("epsilon", 1e-5f))
    , expavgf_(n.attributes_.get("expavg", 0.1f))
    , y_beta_(n.attributes_.get("y.beta", 0.0f))
  {
    mode_ = CUDNN_BATCHNORM_SPATIAL_PERSISTENT;
    bnOps_ = CUDNN_BATCHNORM_OPS_BN_ACTIVATION;

    chkCUDNN(cudnnCreateActivationDescriptor(&desc_));
    chkCUDNN(cudnnSetActivationDescriptor(desc_, CUDNN_ACTIVATION_RELU,
                                          CUDNN_PROPAGATE_NAN, 0.0f));

    size_t workspace;

    chkCUDNN(cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(ctx_->cudnn_,
                                                                      mode_, bnOps_,
                                                                      x_->desc(),
                                                                      NULL,
                                                                      y_->desc(),
                                                                      s_->desc(),
                                                                      desc_,
                                                                      &workspace));
    p.requetstWorkspace(workspace);


    chkCUDNN(cudnnGetBatchNormalizationTrainingExReserveSpaceSize(ctx_->cudnn_,
                                                                  mode_, bnOps_,
                                                                  desc_,
                                                                  x_->desc(),
                                                                  &reserve_size_));
    chkCuda(cudaMalloc(&reserve_, reserve_size_));

  }

  void print() const {
    printf("CudnnBatchNormActivationTrain\n");
    printf("\tx: %s\n", x_->info().c_str());
    printf("\ts: %s\n", s_->info().c_str());
    printf("\tb: %s\n", b_->info().c_str());
    printf("\tm: %s\n", m_->info().c_str());
    printf("\tv: %s\n", v_->info().c_str());
    printf("\ty: %s\n", y_->info().c_str());
  }

  ~CudnnBatchNormActivationTrain()
  {
    chkCUDNN(cudnnDestroyActivationDescriptor(desc_));
    chkCuda(cudaFree(reserve_));
  }


  void exec(CudaProgram &p) {
    float alpha = 1.0f;
    chkCUDNN(cudnnBatchNormalizationForwardTrainingEx(ctx_->cudnn_,
                                                      mode_, bnOps_,
                                                      &alpha, &y_beta_,
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
                                                      p.workspace_, p.workspace_size_,
                                                      reserve_, reserve_size_));
  }
};



struct CudnnBatchNormActivationBwd : public CudaOperation {

  const std::shared_ptr<CudaContext> ctx_;
  const std::shared_ptr<CudnnBatchNormActivationTrain> fwd_;
  const std::shared_ptr<CudaTensor> dy_, dx_, ds_, db_;
  const float dx_beta_;
  CudnnBatchNormActivationBwd(CudaProgram &p, const Node &n,
                              std::shared_ptr<CudnnBatchNormActivationTrain> fwd)
    : ctx_(p.ctx_)
    , fwd_(fwd)
    , dy_(fwd->y_->makeGrad())
    , dx_(fwd->x_->makeGrad())
    , ds_(fwd->s_->makeGrad())
    , db_(fwd->b_->makeGrad())
    , dx_beta_(n.attributes_.get("dx.beta", 0.0f))
  {
    size_t workspace;

    chkCUDNN(cudnnGetBatchNormalizationBackwardExWorkspaceSize(ctx_->cudnn_,
                                                               fwd_->mode_,
                                                               fwd_->bnOps_,
                                                               fwd_->x_->desc(),
                                                               fwd_->y_->desc(),
                                                               dy_->desc(),
                                                               NULL,
                                                               dx_->desc(),
                                                               ds_->desc(),
                                                               fwd->desc_,
                                                               &workspace));
    p.requetstWorkspace(workspace);
  }

  void print() const {
    printf("CudnnBatchNormActivationBwd\n");
  }


  void exec(CudaProgram &p) {
    float alpha = 1.0f, beta = 0.0f;
    chkCUDNN(cudnnBatchNormalizationBackwardEx(ctx_->cudnn_,
                                               fwd_->mode_, fwd_->bnOps_,
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
                                               p.workspace_, p.workspace_size_,
                                               fwd_->reserve_, fwd_->reserve_size_));
  }
};



static void
batchnorm_relu_train(CudaProgram &p, const Node &n)
{
  auto f = std::make_shared<CudnnBatchNormActivationTrain>(p, n);
  p.train(f);

  auto b = std::make_shared<CudnnBatchNormActivationBwd>(p, n, f);
  p.bwd(b);

  p.upd(std::make_shared<CudnnAdam>(p, f->s_, b->ds_));
  p.upd(std::make_shared<CudnnAdam>(p, f->b_, b->db_));
}

REGISTER_CUDA_OP("batchnorm_relu", NULL, batchnorm_relu_train);





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

struct CudnnActivationFwd : public CudaOperation {
  const std::shared_ptr<CudaContext> ctx_;
  const std::shared_ptr<CudaTensor> x_, y_;
  const float y_beta_;

  cudnnActivationDescriptor_t desc_;

  CudnnActivationFwd(CudaProgram &p, const Node &n,
                     cudnnActivationMode_t mode, float alpha)
    : ctx_(p.ctx_)
    , x_(p.lower_tensor_batch(n.inputs_.get("x")))
    , y_(p.lower_tensor_batch(n.outputs_.get("y")))
    , y_beta_(n.attributes_.get("y.beta", 0.0f))
  {
    chkCUDNN(cudnnCreateActivationDescriptor(&desc_));
    chkCUDNN(cudnnSetActivationDescriptor(desc_, mode,
                                          CUDNN_PROPAGATE_NAN, alpha));
  }

  ~CudnnActivationFwd()
  {
    chkCUDNN(cudnnDestroyActivationDescriptor(desc_));
  }

  void print() const {
    printf("Activation Fwd\n");
    printf("\tx: %s\n", x_->info().c_str());
    printf("\ty: %s\n", y_->info().c_str());
  }

  void exec(CudaProgram &p) {
    float alpha = 1.0f;

    chkCUDNN(cudnnActivationForward(ctx_->cudnn_, desc_,
                                    &alpha,
                                    x_->desc(), x_->deviceMem(),
                                    &y_beta_,
                                    y_->desc(), y_->deviceMem()));
  }
};


struct CudnnActivationBwd : public CudaOperation {
  const std::shared_ptr<CudaContext> ctx_;
  const std::shared_ptr<CudnnActivationFwd> fwd_;
  const std::shared_ptr<CudaTensor> dx_, dy_;
  const float dx_beta_;
  cudnnActivationDescriptor_t desc_;

  CudnnActivationBwd(CudaProgram &p, const Node &n,
                     const std::shared_ptr<CudnnActivationFwd> fwd)
    : ctx_(p.ctx_)
    , fwd_(fwd)
    , dx_(fwd->x_->makeGrad())
    , dy_(fwd->y_->makeGrad())
    , dx_beta_(n.attributes_.get("dx.beta", 0.0f))
  {
  }

  ~CudnnActivationBwd()
  {
  }

  void print() const {
    printf("Activation Bwd\n");
    printf("\tdy: %s\n", dy_->info().c_str());
    printf("\tdx: %s\n", dx_->info().c_str());
  }

  void exec(CudaProgram &p) {
    float alpha = 1.0f;

    chkCUDNN(cudnnActivationBackward(ctx_->cudnn_, fwd_->desc_,
                                     &alpha,
                                     fwd_->y_->desc(), fwd_->y_->deviceMem(),
                                     dy_->desc(), dy_->deviceMem(),
                                     fwd_->x_->desc(), fwd_->x_->deviceMem(),
                                     &dx_beta_,
                                     dx_->desc(), dx_->deviceMem()));
  }
};


static void
relu_infer(CudaProgram &p, const Node &n)
{
  p.infer(std::make_shared<CudnnActivationFwd>(p, n, CUDNN_ACTIVATION_RELU,
                                               0.0f));
}

static void
relu_train(CudaProgram &p, const Node &n)
{
  auto f = std::make_shared<CudnnActivationFwd>(p, n, CUDNN_ACTIVATION_RELU,
                                                0.0f);
  p.train(f);
  if(f->x_->grad_)
    p.bwd(std::make_shared<CudnnActivationBwd>(p, n, f));
}

REGISTER_CUDA_OP("relu", relu_infer, relu_train);



static void
elu_infer(CudaProgram &p, const Node &n)
{
  p.infer(std::make_shared<CudnnActivationFwd>(p, n, CUDNN_ACTIVATION_ELU,
                                               n.attributes_.get("alpha", 0.1f)));
}

static void
elu_train(CudaProgram &p, const Node &n)
{
  auto f = std::make_shared<CudnnActivationFwd>(p, n, CUDNN_ACTIVATION_ELU,
                                                n.attributes_.get("alpha", 0.1f));
  p.train(f);
  if(f->x_->grad_)
    p.bwd(std::make_shared<CudnnActivationBwd>(p, n, f));
}

REGISTER_CUDA_OP("elu", elu_infer, elu_train);

//------------------------------------------------------------------------

struct CudnnPoolingFwd : public CudaOperation {

  const std::shared_ptr<CudaContext> ctx_;
  const std::shared_ptr<CudaTensor> x_, y_;
  const float y_beta_;

  cudnnPoolingDescriptor_t desc_;

  CudnnPoolingFwd(CudaProgram &p, const Node &n, cudnnPoolingMode_t mode)
    : ctx_(p.ctx_)
    , x_(p.lower_tensor_batch(n.inputs_.get("x")))
    , y_(p.lower_tensor_batch(n.outputs_.get("y")))
    , y_beta_(n.attributes_.get("y.beta", 0.0f))
  {
    int size;
    if(n.attributes_.get("global", false)) {
      size = x_->dims_[2];
    } else {
      size = n.attributes_.get("size", 1);
    }
    const int pad    = n.attributes_.get("pad", 0);
    const int stride = n.attributes_.get("stride", 1);

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

  void print() const {
    printf("Pooling Fwd\n");
    printf("\tx: %s\n", x_->info().c_str());
    printf("\ty: %s\n", y_->info().c_str());
  }

  void exec(CudaProgram &p) {
    float alpha = 1.0f;

    chkCUDNN(cudnnPoolingForward(ctx_->cudnn_, desc_,
                                 &alpha,
                                 x_->desc(), x_->deviceMem(),
                                 &y_beta_,
                                 y_->desc(), y_->deviceMem()));
  }
};




struct CudnnPoolingBwd : public CudaOperation {

  const std::shared_ptr<CudaContext> ctx_;
  const std::shared_ptr<CudnnPoolingFwd> fwd_;
  const std::shared_ptr<CudaTensor> dx_, dy_;
  const float dx_beta_;

  CudnnPoolingBwd(CudaProgram &p, const Node &n,
                  std::shared_ptr<CudnnPoolingFwd> fwd)
    : ctx_(p.ctx_)
    , fwd_(fwd)
    , dx_(fwd->x_->makeGrad())
    , dy_(fwd->y_->makeGrad())
    , dx_beta_(n.attributes_.get("dx.beta", 0.0f))
  {
  }

  void print() const {
    printf("Pooling Bwd\n");
    printf("\tdy: %s\n", dy_->info().c_str());
    printf("\tdx: %s\n", dx_->info().c_str());
  }

  void exec(CudaProgram &p) {
    float alpha = 1.0f;

    chkCUDNN(cudnnPoolingBackward(ctx_->cudnn_, fwd_->desc_,
                                  &alpha,
                                  fwd_->y_->desc(), fwd_->y_->deviceMem(),
                                  dy_->desc(), dy_->deviceMem(),
                                  fwd_->x_->desc(), fwd_->x_->deviceMem(),
                                  &dx_beta_,
                                  dx_->desc(),
                                  dx_->deviceMem()));

    if(p.debug_) {
      dy_->printStats("pooling.dy");
      if(dx_)
        dx_->printStats("pooling.dx");
    }

  }
};


static void
maxpool_infer(CudaProgram &p, const Node &n)
{
  p.infer(std::make_shared<CudnnPoolingFwd>(p, n, CUDNN_POOLING_MAX));
}

static void
maxpool_train(CudaProgram &p, const Node &n)
{
  auto f = std::make_shared<CudnnPoolingFwd>(p, n, CUDNN_POOLING_MAX);
  p.train(f);
  if(f->x_->grad_)
    p.bwd(std::make_shared<CudnnPoolingBwd>(p, n, f));
}

REGISTER_CUDA_OP("maxpool", maxpool_infer, maxpool_train);

static void
avgpool_infer(CudaProgram &p, const Node &n)
{
  p.infer(std::make_shared<CudnnPoolingFwd>(p, n, CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING));
}

static void
avgpool_train(CudaProgram &p, const Node &n)
{
  auto f = std::make_shared<CudnnPoolingFwd>(p, n, CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING);
  p.train(f);
  if(f->x_->grad_)
    p.bwd(std::make_shared<CudnnPoolingBwd>(p, n, f));
}

REGISTER_CUDA_OP("avgpool", avgpool_infer, avgpool_train);

//------------------------------------------------------------------------

struct CudnnSumFwd : public CudaOperation {

  const std::shared_ptr<CudaContext> ctx_;
  const std::shared_ptr<CudaTensor> x0_, x1_, y_;
  cudnnOpTensorDescriptor_t desc_;

  CudnnSumFwd(CudaProgram &p, const Node &n)
    : ctx_(p.ctx_)
    , x0_(p.lower_tensor_batch(n.inputs_.get("x0")))
    , x1_(p.lower_tensor_batch(n.inputs_.get("x1")))
    , y_(p.lower_tensor_batch(n.outputs_.get("y")))
  {
    cudnnCreateOpTensorDescriptor(&desc_);
    cudnnSetOpTensorDescriptor(desc_,
                               CUDNN_OP_TENSOR_ADD,
                               CUDNN_DATA_FLOAT,
                               CUDNN_PROPAGATE_NAN);
  }

  ~CudnnSumFwd()
  {
    cudnnDestroyOpTensorDescriptor(desc_);
  }

  void print() const {
    printf("Sum\n");
    printf("\ty: %s\n", y_->info().c_str());
  }

  void exec(CudaProgram &p) {

    float alpha = 1.0f;
    float beta = 0.0f;

    cudnnOpTensor(ctx_->cudnn_, desc_,
                  &alpha,
                  x0_->desc(),
                  x0_->deviceMem(),
                  &alpha,
                  x1_->desc(),
                  x1_->deviceMem(),
                  &beta,
                  y_->desc(),
                  y_->deviceMem());
  }
};

static void
sum_infer(CudaProgram &p, const Node &n)
{
  p.infer(std::make_shared<CudnnSumFwd>(p, n));
}

REGISTER_CUDA_OP("sum", sum_infer, NULL);


//------------------------------------------------------------------------
#if 0

static std::shared_ptr<Node>
sum_transform(CudaProgram &p, std::shared_ptr<Node> n)
{
  auto y = p.lower_tensor_batch(n->outputs_.get("y"));
  auto xvec = n->inputs_.getv("x");

  Dims d = y->dims_;
  d.insert(d.begin(), xvec.size());

  int strides[d.size()];
  int dims[d.size()];
  int rank;
  cudnnDataType_t data_type;

  chkCUDNN(cudnnGetTensorNdDescriptor(y->desc_, y->dims_.size(),
                                      &data_type,
                                      &rank, dims + 1, strides + 1));

  strides[0] = xvec.size() * strides[1];
  auto t = std::make_shared<CudaTensor>(y->data_type_, d, (const int *)strides,
                                        p.ctx_,
                                        y->namePostfix("sum"));

  int64_t offset = 0;
  for(const auto &xh : xvec) {
    p.tensors_[xh] = std::make_shared<CudaTensor>(t->storage_,
                                                  xh->dims_,
                                                  offset,
                                                  (const int *)strides + 1,
                                                  p.ctx_,
                                                  xh->namePostfix("sum.alias"));
    offset += strides[0];
  }

  auto nn = std::make_shared<Node>("cudnn.reduce.add");
  p.tensors_[t] = t;

  d[0] = 1;
  auto ya = std::make_shared<CudaTensor>(y->storage_,
                                         d,
                                         0,
                                         (const int *)strides,
                                         p.ctx_,
                                         y->namePostfix("sum"));

  p.tensors_[ya] = ya;
  nn->inputs_["x"] = t;
  nn->outputs_["y"] = ya;
  return nn;
}


//------------------------------------------------------------------------

struct CudnnReduce : public CudaOperation {

  const std::shared_ptr<CudaContext> ctx_;
  const std::shared_ptr<CudaTensor> x_, y_;
  cudnnReduceTensorDescriptor_t desc_;

  CudnnReduce(CudaProgram &p,
              std::shared_ptr<CudaTensor> x,
              std::shared_ptr<CudaTensor> y,
              cudnnReduceTensorOp_t op)
    : ctx_(p.ctx_)
    , x_(x)
    , y_(y)
  {
    chkCUDNN(cudnnCreateReduceTensorDescriptor(&desc_));
    chkCUDNN(cudnnSetReduceTensorDescriptor(desc_, op,
                                            x->type_,
                                            CUDNN_PROPAGATE_NAN,
                                            CUDNN_REDUCE_TENSOR_NO_INDICES,
                                            CUDNN_32BIT_INDICES));

    size_t workspace;

    chkCUDNN(cudnnGetReductionWorkspaceSize(ctx_->cudnn_, desc_,
                                            x_->desc_,
                                            y_->desc_,
                                            &workspace));
    p.requetstWorkspace(workspace);
  }

  ~CudnnReduce()
  {
    chkCUDNN(cudnnDestroyReduceTensorDescriptor(desc_));
  }

  void print() const {
    printf("Reduce\n");
    printf("\tx: %s\n", x_->info().c_str());
    printf("\ty: %s\n", y_->info().c_str());
  }

  void exec(CudaProgram &p) {

    float alpha = 1.0f, beta = 0.0f;

    chkCUDNN(cudnnReduceTensor(ctx_->cudnn_,
                               desc_,
                               NULL, 0,
                               p.workspace_, p.workspace_size_,
                               &alpha,
                               x_->desc(),
                               x_->deviceMem(),
                               &beta,
                               y_->desc(),
                               y_->deviceMem()));
  }
};


static void
cudnn_reduce_add_make(CudaProgram &p, const Node &n)
{
  auto f = std::make_shared<CudnnReduce>(p,
                                         p.lower_tensor(n.inputs_.get("x")),
                                         p.lower_tensor(n.outputs_.get("y")),
                                         CUDNN_REDUCE_TENSOR_ADD);
  p.fwd(f);
}
 #endif

//------------------------------------------------------------------------

struct CudnnGemmFwd : public CudaOperation {

  const std::shared_ptr<CudaContext> ctx_;
  const std::shared_ptr<CudaTensor> x_, w_, b_, y_;
  const int n_;
  const int num_inputs_;
  const int num_outputs_;
  const bool transW_;
  const float y_beta_;

  CudnnGemmFwd(CudaProgram &p, const Node &n)
    : ctx_(p.ctx_)
    , x_(p.lower_tensor_batch(n.inputs_.get("x")))
    , w_(p.lower_tensor(n.inputs_.get("w")))
    , b_(p.lower_tensor(n.inputs_.get("b"), 2))
    , y_(p.lower_tensor_batch(n.outputs_.get("y")))
    , n_(x_->dims_[0])
    , num_inputs_(x_->dims_[1])
    , num_outputs_(y_->dims_[1])
    , transW_(n.attributes_.get("transW", false))
    , y_beta_(n.attributes_.get("y.beta", 0.0f))
  {
    assert(y_beta_ == 0); // The trailing cudnnAddTensor operations doesn't support this
  }

  void print() const {
    printf("Gemm Fwd (%d inputs, %d outputs)\n",
           num_inputs_, num_outputs_);
    printf("\tx: %s\n", x_->info().c_str());
    printf("\tw: %s\n", w_->info().c_str());
    if(b_)
      printf("\tb: %s\n", b_->info().c_str());
    printf("\ty: %s\n", y_->info().c_str());
  }

  void exec(CudaProgram &p) {

    float alpha = 1.0f, beta = 0.0f;
    __half halpha = 1.0f, hbeta = 0.0f;
    cublasOperation_t transA = transW_ ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t transB = CUBLAS_OP_N;
    switch(x_->type_) {
    case CUDNN_DATA_FLOAT:
      chkCuda(cublasSgemm(ctx_->cublas_, transA, transB,
                          num_outputs_, n_, num_inputs_,
                          &alpha,
                          (const float *)w_->deviceMem(), transW_ ? num_inputs_ : num_outputs_,
                          (const float *)x_->deviceMem(), num_inputs_,
                          &beta,
                          (float *)y_->deviceMem(), num_outputs_));
      break;
    case CUDNN_DATA_HALF:
      chkCuda(cublasHgemm(ctx_->cublas_, transA, transB,
                          num_outputs_, n_, num_inputs_,
                          &halpha,
                          (const __half *)w_->deviceMem(), transW_ ? num_inputs_ : num_outputs_,
                          (const __half *)x_->deviceMem(), num_inputs_,
                          &hbeta,
                          (__half *)y_->deviceMem(), num_outputs_));
      break;
    default:
      abort();
    }
    if(b_) {
      chkCUDNN(cudnnAddTensor(ctx_->cudnn_,
                              &alpha, b_->desc(), b_->deviceMem(),
                              &alpha, y_->desc(), y_->deviceMem()));
    }

    if(p.debug_) {
      x_->printStats("gemm.x");
      w_->printStats("gemm.w");
      b_->printStats("gemm.b");
      y_->printStats("gemm.y");
    }


  }
};


struct CudnnGemmBwd : public CudaOperation {

  const std::shared_ptr<CudaContext> ctx_;
  const std::shared_ptr<CudaTensor> dx_, dw_, db_, dy_, x_, w_;
  const int n_;
  const int num_inputs_;
  const int num_outputs_;
  const std::shared_ptr<CudaTensor> ones_;
  const float dx_beta_;

  CudnnGemmBwd(CudaProgram &p, const Node &n,
               std::shared_ptr<CudnnGemmFwd> fwd)
    : ctx_(p.ctx_)
    , dx_(fwd->x_->makeGrad())
    , dw_(fwd->w_->makeGrad())
    , db_(fwd->b_->makeGrad())
    , dy_(fwd->y_->makeGrad())
    , x_(fwd->x_)
    , w_(fwd->w_)
    , n_(fwd->n_)
    , num_inputs_(fwd->num_inputs_)
    , num_outputs_(fwd->num_outputs_)
    , ones_(p.lower_tensor(Tensor::make(x_->data_type_, {n_,1}, 1, 0)))
    , dx_beta_(n.attributes_.get("dx.beta", 0.0f))
  {
    assert(fwd->transW_ == true);
  }

  void print() const {
    printf("Gemm Bwd\n");
    printf("\tdy: %s\n", dy_->info().c_str());
    printf("\tdb: %s\n", db_->info().c_str());
    printf("\tdw: %s\n", dw_->info().c_str());
    if(dx_)
      printf("\tdx: %s\n", dx_->info().c_str());
  }

  void exec(CudaProgram &p) {
    float alpha = 1.0f, beta = 0.0f;
    __half halpha = 1.0f, hbeta = 0.0f;

    switch(x_->type_) {
    case CUDNN_DATA_FLOAT:
      chkCuda(cublasSgemm(ctx_->cublas_, CUBLAS_OP_N, CUBLAS_OP_T,
                          num_inputs_, num_outputs_, n_,
                          &alpha,
                          (const float *)x_->deviceMem(), num_inputs_,
                          (const float *)dy_->deviceMem(),
                          num_outputs_,
                          &beta,
                          (float *)dw_->deviceMem(), num_inputs_));

      chkCuda(cublasSgemv(ctx_->cublas_, CUBLAS_OP_N, num_outputs_, n_,
                          &alpha,
                          (const float *)dy_->deviceMem(), num_outputs_,
                          (const float *)ones_->deviceMem(), 1,
                          &beta,
                          (float *)db_->deviceMem(), 1));

      if(dx_ != NULL) {
        chkCuda(cublasSgemm(ctx_->cublas_, CUBLAS_OP_N, CUBLAS_OP_N,
                            num_inputs_, n_, num_outputs_,
                            &alpha,
                            (const float *)w_->deviceMem(), num_inputs_,
                            (const float *)dy_->deviceMem(),
                            num_outputs_,
                            &dx_beta_,
                            (float *)dx_->deviceMem(), num_inputs_));
      }

      break;

    case CUDNN_DATA_HALF:
      chkCuda(cublasHgemm(ctx_->cublas_, CUBLAS_OP_N, CUBLAS_OP_T,
                          num_inputs_, num_outputs_, n_,
                          &halpha,
                          (const __half *)x_->deviceMem(), num_inputs_,
                          (const __half *)dy_->deviceMem(),
                          num_outputs_,
                          &hbeta,
                          (__half *)dw_->deviceMem(), num_inputs_));

      // No cublasSgemv() for half type, so do matrix*matrix instead
      chkCuda(cublasHgemm(ctx_->cublas_, CUBLAS_OP_N, CUBLAS_OP_T,
                          1, num_outputs_, n_,
                          &halpha,
                          (const __half *)ones_->deviceMem(),
                          1,
                          (const __half *)dy_->deviceMem(),
                          num_outputs_,
                          &hbeta,
                          (__half *)db_->deviceMem(), 1));

      if(dx_ != NULL) {
        __half dx_hbeta = dx_beta_;
        chkCuda(cublasHgemm(ctx_->cublas_, CUBLAS_OP_N, CUBLAS_OP_N,
                            num_inputs_, n_, num_outputs_,
                            &halpha,
                            (const __half *)w_->deviceMem(), num_inputs_,
                            (const __half *)dy_->deviceMem(),
                            num_outputs_,
                            &dx_hbeta,
                            (__half *)dx_->deviceMem(), num_inputs_));
      }
      break;

    default:
      abort();
    }

    if(p.debug_) {
      x_->print("gemm.x", 4);
      w_->print("gemm.w", 4);
      dy_->print("gemm.dy", 4);
      dw_->print("gemm.dw", 4);
      db_->print("gemm.db", 4);
      if(dx_)
        dx_->print("gemm.dx", 4);
    }
  }
};



static void
fc_infer(CudaProgram &p, const Node &n)
{
  p.infer(std::make_shared<CudnnGemmFwd>(p, n));
}

static void
fc_train(CudaProgram &p, const Node &n)
{
  auto f = std::make_shared<CudnnGemmFwd>(p, n);
  p.train(f);
  auto b = std::make_shared<CudnnGemmBwd>(p, n, f);
  p.bwd(b);

  p.upd(std::make_shared<CudnnAdam>(p, f->w_, b->dw_));
  if(f->b_)
    p.upd(std::make_shared<CudnnAdam>(p, f->b_, b->db_));
}

REGISTER_CUDA_OP("fc", fc_infer, fc_train);


//------------------------------------------------------------------------

struct CudnnSoftmaxFwd : public CudaOperation {

  const std::shared_ptr<CudaContext> ctx_;
  const std::shared_ptr<CudaTensor> x_, y_;

  CudnnSoftmaxFwd(CudaProgram &p, const Node &n)
    : ctx_(p.ctx_)
    , x_(p.lower_tensor_batch(n.inputs_.get("x")))
    , y_(p.lower_tensor_batch(n.outputs_.get("y")))
  {
  }

  void print() const {
    printf("Softmax Fwd\n");
    printf("\tx: %s\n", x_->info().c_str());
    printf("\ty: %s\n", y_->info().c_str());
  }

  void exec(CudaProgram &p) {

    float alpha = 1.0f, beta = 0.0f;

    chkCUDNN(cudnnSoftmaxForward(ctx_->cudnn_,
                                 CUDNN_SOFTMAX_ACCURATE,
                                 CUDNN_SOFTMAX_MODE_CHANNEL,
                                 &alpha,
                                 x_->desc(), x_->deviceMem(),
                                 &beta,
                                 y_->desc(), y_->deviceMem()));
  }
};

static void
softmax_infer(CudaProgram &p, const Node &n)
{
  p.infer(std::make_shared<CudnnSoftmaxFwd>(p, n));
}

REGISTER_CUDA_OP("softmax", softmax_infer, NULL);


//------------------------------------------------------------------------
struct CudnnCatClassifierFwd : public CudaOperation {

  const std::shared_ptr<CudaContext> ctx_;
  const std::shared_ptr<CudaTensor> x_, y_;

  CudnnCatClassifierFwd(CudaProgram &p, const Node &n)
    : ctx_(p.ctx_)
    , x_(p.lower_tensor_batch(n.inputs_.get("x")))
    , y_(p.lower_tensor_batch(n.outputs_.get("y")))
  {
  }

  void print() const {
    printf("CatClassifier Fwd\n");
    printf("\tx: %s\n", x_->info().c_str());
    printf("\ty: %s\n", y_->info().c_str());
  }

  void exec(CudaProgram &p) {
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
      abort();
    }
  }
};

struct CudnnCatClassifierBwd : public CudaOperation {

  const std::shared_ptr<CudaContext> ctx_;
  const std::shared_ptr<CudnnCatClassifierFwd> fwd_;
  const std::shared_ptr<CudaTensor> dx_, dy_, loss_;

  CudnnCatClassifierBwd(CudaProgram &p, const Node &n,
                        std::shared_ptr<CudnnCatClassifierFwd> fwd)
    : ctx_(p.ctx_)
    , fwd_(fwd)
    , dx_(fwd->x_->makeGrad())
    , dy_(fwd->y_->makeGrad())
    , loss_(p.lower_tensor_batch(n.outputs_.get("loss")))
  {
  }

  void print() const {
    printf("CatClassifier Bwd\n");
    printf("\tdy: %s\n", dy_->info().c_str());
    printf("\tdx: %s\n", dx_->info().c_str());
  }

  void exec(CudaProgram &p) {

    const int n = fwd_->x_->dims_[0];
    const int c = fwd_->x_->dims_[1];
    const float scale = 1.0f / n;

    switch(fwd_->x_->type_) {
    case CUDNN_DATA_FLOAT:
      catclassifier_bwd_float_i32(n,
                                  (const float *)fwd_->x_->deviceMem(),
                                  (float *)dx_->deviceMem(),
                                  (const int32_t *)fwd_->y_->deviceMem(),
                                  (const int32_t *)dy_->deviceMem(),
                                  loss_ ? (float *)loss_->deviceMem() : NULL,
                                  c, scale,
                                  p.ctx_->stream_);

      break;
    case CUDNN_DATA_HALF:
      catclassifier_bwd_half_i32(n,
                                 (const __half *)fwd_->x_->deviceMem(),
                                 (__half *)dx_->deviceMem(),
                                 (const int32_t *)fwd_->y_->deviceMem(),
                                 (const int32_t *)dy_->deviceMem(),
                                 loss_ ? (float *)loss_->deviceMem() : NULL,
                                 c, scale * p.mp_scaling_,
                                 p.ctx_->stream_);

      break;
    default:
      abort();
    }

    if(p.debug_) {
      fwd_->x_->print("catclassifier.x");
      fwd_->y_->print("catclassifier.y");
      dy_->print("catclassifier.dy");
      dx_->print("catclassifier.dx");
    }

  }

};

static void
catclassifier_infer(CudaProgram &p, const Node &n)
{
  p.infer(std::make_shared<CudnnCatClassifierFwd>(p, n));
}

static void
catclassifier_train(CudaProgram &p, const Node &n)
{
  auto f = std::make_shared<CudnnCatClassifierFwd>(p, n);
  p.train(f);
  p.bwd(std::make_shared<CudnnCatClassifierBwd>(p, n, f));
}

REGISTER_CUDA_OP("catclassifier", catclassifier_infer, catclassifier_train);


//------------------------------------------------------------------------

struct CudnnTransform : public CudaOperation {

  const std::shared_ptr<CudaContext> ctx_;
  const std::shared_ptr<CudaTensor> a_, b_;
  const float beta_;

  CudnnTransform(CudaProgram &p,
                 std::shared_ptr<CudaTensor> a,
                 std::shared_ptr<CudaTensor> b,
                 float beta)
    : ctx_(p.ctx_)
    , a_(a)
    , b_(b)
    , beta_(beta)
  {}

  void print() const {
    printf("Transform\n");
    printf("\ta: %s\n", a_->info().c_str());
    printf("\tb: %s\n", b_->info().c_str());
  }

  void exec(CudaProgram &p) {
    float alpha = 1.0f;
    chkCUDNN(cudnnTransformTensor(ctx_->cudnn_,
                                  &alpha,
                                  a_->desc(),
                                  a_->deviceMem(),
                                  &beta_,
                                  b_->desc(),
                                  b_->deviceMem()));
  }
};

//------------------------------------------------------------------------

static void
concat_transform_node(CudaProgram &p, const Node &n)
{
  const int axis = 1;

  auto y = p.lower_tensor_batch(n.outputs_.get("y"));
  auto dy = y->makeGrad();
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


//------------------------------------------------------------------------



struct CudnnConvert : public CudaOperation {

  const std::shared_ptr<CudaContext> ctx_;
  const std::shared_ptr<CudaTensor> x_, y_;
  const float scale_;
  void (*algo_)(const void *src, void *dst, int elements, float scale,
                cudaStream_t stream);

  CudnnConvert(CudaProgram &p,
               std::shared_ptr<CudaTensor> x,
               std::shared_ptr<CudaTensor> y,
               float scale)
    : ctx_(p.ctx_)
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

  void print() const {
    printf("Convert %zd elements\n", (size_t)x_->elements_);
    printf("\tx: %s\n", x_->info().c_str());
    printf("\ty: %s\n", y_->info().c_str());
  }

  void exec(CudaProgram &p) {
    algo_(x_->deviceMem(), y_->deviceMem(), x_->elements_, scale_,
          p.ctx_->stream_);
  }

};



static void
convert_infer(CudaProgram &p, const Node &n)
{
  auto x = p.lower_tensor_batch(n.inputs_.get("x"));
  auto y = p.lower_tensor_batch(n.outputs_.get("y"), *x);
  auto scale = n.attributes_.get("scale", 1.0f);
  p.infer(std::make_shared<CudnnConvert>(p, x, y, scale));
}

static void
convert_train(CudaProgram &p, const Node &n)
{
  auto x = p.lower_tensor_batch(n.inputs_.get("x"));
  auto y = p.lower_tensor_batch(n.outputs_.get("y"), *x);
  auto scale = n.attributes_.get("scale", 1.0f);
  p.train(std::make_shared<CudnnConvert>(p, x, y, scale));

  assert(y->grad_ == NULL); // No backprop here yet
}

REGISTER_CUDA_OP("convert", convert_infer, convert_train);


//------------------------------------------------------------------------

struct CudnnMathOp : public CudaOperation {

  const std::shared_ptr<CudaContext> ctx_;
  const std::shared_ptr<CudaTensor> a_, b_, c_;
  cudnnOpTensorDescriptor_t desc_;

  CudnnMathOp(CudaProgram &p,
              std::shared_ptr<CudaTensor> a,
              std::shared_ptr<CudaTensor> b,
              std::shared_ptr<CudaTensor> c,
              cudnnOpTensorOp_t op)
    : ctx_(p.ctx_)
    , a_(a)
    , b_(b)
    , c_(c)
  {
    chkCUDNN(cudnnCreateOpTensorDescriptor(&desc_));
    chkCUDNN(cudnnSetOpTensorDescriptor(desc_,
                                        op, c->type_,
                                        CUDNN_PROPAGATE_NAN));
  }

  ~CudnnMathOp()
  {
    chkCUDNN(cudnnDestroyOpTensorDescriptor(desc_));
  }

  void print() const {
    printf("MathOp\n");
    printf("\ta: %s\n", a_->info().c_str());
    printf("\tb: %s\n", b_->info().c_str());
    printf("\tc: %s\n", c_->info().c_str());
  }

  void exec(CudaProgram &p) {

    float alpha = 1.0f, beta = 0.0f;

    chkCUDNN(cudnnOpTensor(ctx_->cudnn_,
                           desc_,
                           &alpha,
                           a_->desc(),
                           a_->deviceMem(),
                           &alpha,
                           b_->desc(),
                           b_->deviceMem(),
                           &beta,
                           c_->desc(),
                           c_->deviceMem()));
  }
};


static void
add_infer(CudaProgram &p, const Node &n)
{
  auto f = std::make_shared<CudnnMathOp>(p,
                                         p.lower_tensor_batch(n.outputs_.get("x")),
                                         p.lower_tensor(n.outputs_.get("b")),
                                         p.lower_tensor_batch(n.outputs_.get("y")),
                                         CUDNN_OP_TENSOR_ADD);
  p.infer(f);
}


static void
mul_infer(CudaProgram &p, const Node &n)
{
  auto f = std::make_shared<CudnnMathOp>(p,
                                         p.lower_tensor_batch(n.outputs_.get("x")),
                                         p.lower_tensor(n.outputs_.get("s")),
                                         p.lower_tensor_batch(n.outputs_.get("y")),
                                         CUDNN_OP_TENSOR_MUL);
  p.infer(f);
}


REGISTER_CUDA_OP("add", add_infer, NULL);
REGISTER_CUDA_OP("mul", mul_infer, NULL);



//------------------------------------------------------------------------


static std::vector<std::shared_ptr<Node>>
reshape_transform_node(CudaProgram &p, std::shared_ptr<Node> n)
{
  auto x = p.lower_tensor_batch(n->inputs_.get("x"), CUDNN_TENSOR_NCHW);
  auto dx = x->makeGrad();
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


struct CudnnDropoutFwd : public CudaOperation {
  const std::shared_ptr<CudaContext> ctx_;
  const std::shared_ptr<CudaTensor> x_, y_;
  const float y_beta_;
  cudnnDropoutDescriptor_t desc_;
  size_t reserve_size_;
  void *reserve_;
  size_t states_size_;
  void *states_;

  CudnnDropoutFwd(CudaProgram &p, const Node &n)
    : ctx_(p.ctx_)
    , x_(p.lower_tensor_batch(n.inputs_.get("x")))
    , y_(p.lower_tensor_batch(n.outputs_.get("y")))
    , y_beta_(n.attributes_.get("y.beta", 0.0f))
  {
    assert(y_beta_ == 0);
    const float prob = n.attributes_.get("prob", 0.5f);
    chkCUDNN(cudnnDropoutGetReserveSpaceSize(x_->desc(), &reserve_size_));
    chkCuda(cudaMalloc(&reserve_, reserve_size_));
    chkCUDNN(cudnnDropoutGetStatesSize(ctx_->cudnn_, &states_size_));
    chkCuda(cudaMalloc(&states_, states_size_));

    chkCUDNN(cudnnCreateDropoutDescriptor(&desc_));
    chkCUDNN(cudnnSetDropoutDescriptor(desc_, ctx_->cudnn_, prob,
                                       states_, states_size_, 0));
  }

  ~CudnnDropoutFwd()
  {
    chkCUDNN(cudnnDestroyDropoutDescriptor(desc_));
  }

  void print() const {
    printf("Dropout Fwd\n");
    printf("\tx: %s\n", x_->info().c_str());
    printf("\ty: %s\n", y_->info().c_str());
  }

  void exec(CudaProgram &p) {
    chkCUDNN(cudnnDropoutForward(ctx_->cudnn_, desc_,
                                 x_->desc(), x_->deviceMem(),
                                 y_->desc(), y_->deviceMem(),
                                 reserve_, reserve_size_));
  }
};


struct CudnnDropoutBwd : public CudaOperation {
  const std::shared_ptr<CudaContext> ctx_;
  const std::shared_ptr<CudnnDropoutFwd> fwd_;
  const std::shared_ptr<CudaTensor> dx_, dy_;
  const float dx_beta_;

  CudnnDropoutBwd(CudaProgram &p, const Node &n,
                     const std::shared_ptr<CudnnDropoutFwd> fwd)
    : ctx_(p.ctx_)
    , fwd_(fwd)
    , dx_(fwd->x_->makeGrad())
    , dy_(fwd->y_->makeGrad())
    , dx_beta_(n.attributes_.get("dx.beta", 0.0f))
  {
    assert(dx_beta_ == 0);
  }

  ~CudnnDropoutBwd()
  {
  }

  void print() const {
    printf("Dropout Bwd\n");
    printf("\tdy: %s\n", dy_->info().c_str());
    printf("\tdx: %s\n", dx_->info().c_str());
  }

  void exec(CudaProgram &p) {
    chkCUDNN(cudnnDropoutBackward(ctx_->cudnn_, fwd_->desc_,
                                  dy_->desc(), dy_->deviceMem(),
                                  dx_->desc(), dx_->deviceMem(),
                                  fwd_->reserve_, fwd_->reserve_size_));
  }
};



static void
dropout_train(CudaProgram &p, const Node &n)
{
  auto f = std::make_shared<CudnnDropoutFwd>(p, n);
  p.train(f);
  if(f->x_->grad_)
    p.bwd(std::make_shared<CudnnDropoutBwd>(p, n, f));
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

REGISTER_CUDA_OP("dropout", NULL, dropout_train);




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

struct CudnnSpatialTransformFwd : public CudaOperation {
  const std::shared_ptr<CudaContext> ctx_;
  const std::shared_ptr<CudaTensor> x_, theta_, y_, grid_;
  const float y_beta_;
  cudnnSpatialTransformerDescriptor_t desc_;

  CudnnSpatialTransformFwd(CudaProgram &p, const Node &n)
    : ctx_(p.ctx_)
    , x_(p.lower_tensor_batch(n.inputs_.get("x")))
    , theta_(p.lower_tensor(n.inputs_.get("theta")))
    , y_(p.lower_tensor_batch(n.outputs_.get("y")))
    , grid_(std::make_shared<CudaTensor>(Tensor::DataType::FLOAT,
                                         Dims{ y_->dims_[0], 2, y_->dims_[2], y_->dims_[3]},
                                         CUDNN_TENSOR_NHWC,
                                         p.ctx_))
    , y_beta_(n.attributes_.get("y.beta", 0.0f))
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

  void print() const {
    printf("SpatialTransform Fwd\n");
    printf("\tx: %s\n", x_->info().c_str());
    printf("\ty: %s\n", y_->info().c_str());
  }

  void exec(CudaProgram &p) {
    float alpha = 1.0f;

    chkCUDNN(cudnnSpatialTfGridGeneratorForward(ctx_->cudnn_, desc_,
                                                theta_->deviceMem(),
                                                grid_->deviceMem()));
    chkCUDNN(cudnnSpatialTfSamplerForward(ctx_->cudnn_, desc_,
                                          &alpha,
                                          x_->desc(), x_->deviceMem(),
                                          grid_->deviceMem(),
                                          &y_beta_,
                                          y_->desc(), y_->deviceMem()));
  }
};



static void
spatialtransform_train(CudaProgram &p, const Node &n)
{
  p.train(std::make_shared<CudnnSpatialTransformFwd>(p, n));
}

static void
spatialtransform_infer(CudaProgram &p, const Node &n)
{
  // FIXME: Replace by skipping node
  auto x = p.lower_tensor_batch(n.inputs_.get("x"));
  auto y = p.lower_tensor_batch(n.outputs_.get("y"));

  p.infer(std::make_shared<CudnnTransform>(p, x, y, 0.0f));
}


REGISTER_CUDA_OP("spatialtransform", spatialtransform_infer,
                 spatialtransform_train);



}

