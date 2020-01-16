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
#include "saga.h"
#include "tensor.h"

#include "cuda_common.h"
#include "cuda_tensor.h"
#include "cuda_kernels.h"

namespace saga {

class CudnnContext : public Context,
                     public std::enable_shared_from_this<CudnnContext> {
public:
  CudnnContext()
    : cudnn_(NULL)
    , cublas_(NULL)
  {}

  ~CudnnContext()
  {}

  int init();

  std::shared_ptr<Program> createProgram(const Graph &graph,
                                         ProgramType type,
                                         int batch_size,
                                         float learning_rate,
                                         TensorLayout layout);

  cudnnHandle_t cudnn_;
  cublasHandle_t cublas_;
};


int
CudnnContext::init()
{
  int device;
  cudaGetDevice(&device);

  struct cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);

  printf("Device name: %s\n", prop.name);
  printf("Compute capability: %d.%d\n", prop.major, prop.minor);
  printf("CanMapHostMem: %d\n", prop.canMapHostMemory);
  printf("ComputeMode: %d\n", prop.computeMode);
  printf("Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
  printf("Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
  printf("Peak Memory Bandwidth (GB/s): %f\n\n",
         2.0 * prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);

  chkCUDNN(cudnnCreate(&cudnn_));
  chkCuda(cublasCreate(&cublas_));

  chkCuda(cublasSetMathMode(cublas_, CUBLAS_TENSOR_OP_MATH));
  return 0;
}



std::shared_ptr<Context> createContext()
{
  auto ctx = std::make_shared<CudnnContext>();

  if(ctx->init())
    return nullptr;

  return ctx;
}


//------------------------------------------------------------------------

class CudnnOperation;

class CudnnProgram : public Program {
public:

  CudnnProgram(std::shared_ptr<CudnnContext> ctx,
               cudnnTensorFormat_t tensor_format,
               ProgramType type,
               int batch_size,
               float learning_rate)
    : ctx_(ctx)
    , tensor_format_(tensor_format)
    , type_(type)
    , batch_size_(batch_size)
    , learning_rate_(learning_rate)
    , debug_(false)
    , workspace_(NULL)
    , workspace_size_(0)
    , workspace_requested_(0)
  {
  }

  ~CudnnProgram()
  {
    chkCuda(cudaFree(workspace_));
  }


  void exec(bool learn);

  void print() const;
  void debug(bool);

  std::shared_ptr<Tensor> resolveTensor(std::shared_ptr<Tensor> t);

  void requetstWorkspace(size_t size) {
    workspace_requested_ = std::max(workspace_requested_, size);
  }

  void allocWorkspace() {
    if(workspace_requested_ <= workspace_size_)
      return;
    workspace_size_ = workspace_requested_;
    chkCuda(cudaFree(workspace_));
    chkCuda(cudaMalloc(&workspace_, workspace_size_));
  }

  const std::shared_ptr<CudnnContext> ctx_;
  const cudnnTensorFormat_t tensor_format_;
  const ProgramType type_;
  const int batch_size_;
  const float learning_rate_;
  bool debug_;

  std::unordered_map<std::shared_ptr<Tensor>,
                     std::shared_ptr<CudaTensor>> tensors_;

  std::vector<std::shared_ptr<CudnnOperation>> infer_operations_;
  std::vector<std::shared_ptr<CudnnOperation>> train_operations_;
  std::vector<std::shared_ptr<CudnnOperation>> bwd_operations_;
  std::vector<std::shared_ptr<CudnnOperation>> upd_operations_;

  void *workspace_;
  size_t workspace_size_;
  size_t workspace_requested_;

  std::shared_ptr<CudaTensor> lower_tensor(std::shared_ptr<Tensor> src,
                                           size_t dimensions = 0);

  std::shared_ptr<CudaTensor> lower_tensor_batch(std::shared_ptr<Tensor> src,
                                                 cudnnTensorFormat_t format);

  std::shared_ptr<CudaTensor> lower_tensor_batch(std::shared_ptr<Tensor> src);

  void fwd(const std::shared_ptr<CudnnOperation> &op)
  {
    infer_operations_.push_back(op);
    if(type_ == ProgramType::INFERENCE)
      return;
    train_operations_.push_back(op);
  }

  void infer(const std::shared_ptr<CudnnOperation> &op)
  {
    infer_operations_.push_back(op);
  }

  void train(const std::shared_ptr<CudnnOperation> &op)
  {
    train_operations_.push_back(op);
  }

  void bwd(const std::shared_ptr<CudnnOperation> &op)
  {
    bwd_operations_.insert(bwd_operations_.begin(), op);
  }

  void upd(const std::shared_ptr<CudnnOperation> &op)
  {
    upd_operations_.push_back(op);
  }

};

std::shared_ptr<Tensor>
CudnnProgram::resolveTensor(std::shared_ptr<Tensor> src)
{
  if(src == nullptr)
    return nullptr;

  auto it = tensors_.find(src);
  if(it != tensors_.end()) {
    return it->second;
  }
  return nullptr;
}


std::shared_ptr<CudaTensor>
CudnnProgram::lower_tensor(std::shared_ptr<Tensor> src,
                           size_t dimensions)
{
  if(src == nullptr)
    return nullptr;

  auto it = tensors_.find(src);
  if(it != tensors_.end()) {
    return it->second;
  }

  std::vector<int64_t> dims = src->dims_;

  if(dimensions) {
    while(dims.size() < dimensions)
      dims.insert(dims.begin(), 1);
  }

  auto t = std::make_shared<CudaTensor>(src->data_type_,
                                        dims, tensor_format_,
                                        src->name_);

  t->copyFrom(*src);
  tensors_[src] = t;
  return t;
}


std::shared_ptr<CudaTensor>
CudnnProgram::lower_tensor_batch(std::shared_ptr<Tensor> src,
                                 cudnnTensorFormat_t tensor_format)
{
  if(src == nullptr)
    return nullptr;

  auto it = tensors_.find(src);
  if(it != tensors_.end()) {
    return it->second;
  }

  std::vector<int64_t> dims = src->dims_;

  dims[0] = batch_size_;

  auto t = std::make_shared<CudaTensor>(src->data_type_,
                                        dims, tensor_format,
                                        src->name_);

  t->copyFrom(*src);
  tensors_[src] = t;
  return t;
}


std::shared_ptr<CudaTensor>
CudnnProgram::lower_tensor_batch(std::shared_ptr<Tensor> src)
{
  return lower_tensor_batch(src, tensor_format_);
}







class CudnnOperation {
public:
  virtual ~CudnnOperation() {}
  virtual void exec(CudnnProgram &p) = 0;
  virtual void print() const = 0;
};

//------------------------------------------------------------------------


void
CudnnProgram::exec(bool learn)
{
  allocWorkspace();

  if(!learn) {

    for(const auto &op : infer_operations_) {
      op->exec(*this);
    }

  } else {

    for(const auto &op : train_operations_) {
      op->exec(*this);
    }
    for(const auto &op : bwd_operations_) {
      op->exec(*this);
    }
    for(const auto &op : upd_operations_) {
      op->exec(*this);
    }
  }
}


void
CudnnProgram::print() const
{
  printf("Inference:\n");
  for(const auto &op : train_operations_) {
    op->print();
  }

  printf("Training:\n");
  for(const auto &op : train_operations_) {
    op->print();
  }
  for(const auto &op : bwd_operations_) {
    op->print();
  }
  for(const auto &op : upd_operations_) {
    op->print();
  }
}

void
CudnnProgram::debug(bool on)
{
  debug_ = on;
}


//------------------------------------------------------------------------
struct CudnnAdam : public CudnnOperation {

  const std::shared_ptr<CudaTensor> weights_, gradient_;
  float learning_rate_;
  float *temp_;
  int iter_;

  CudnnAdam(CudnnProgram &p,
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

  void exec(CudnnProgram &p) {
    const int i = ++iter_;

#define ADAM_B1      0.9
#define ADAM_B2      0.999

    const float b1t = 1.0 / (1.0 - pow(ADAM_B1, i));
    const float b2t = 1.0 / (1.0 - pow(ADAM_B2, i));

    switch(weights_->data_type_) {
    case Tensor::DataType::FLOAT:
      adam_float(weights_->elements_, learning_rate_,
                 (float *)weights_->deviceMem(),
                 (const float *)gradient_->deviceMem(),
                 (float *)temp_, b1t, b2t);
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


struct CudnnConvolutionFwd : public CudnnOperation {

  const std::shared_ptr<CudnnContext> ctx_;
  const std::shared_ptr<CudaTensor> x_, w_, b_, y_;

  cudnnConvolutionDescriptor_t conv_desc_;
  cudnnFilterDescriptor_t filter_desc_;
  cudnnConvolutionFwdAlgo_t conv_fwd_algo_;

  ~CudnnConvolutionFwd()
  {
    chkCUDNN(cudnnDestroyFilterDescriptor(filter_desc_));
    chkCUDNN(cudnnDestroyConvolutionDescriptor(conv_desc_));
  }

  CudnnConvolutionFwd(CudnnProgram &p, const Node &n)
    : ctx_(p.ctx_)
    , x_(p.lower_tensor_batch(n.inputs_.get("x")))
    , w_(p.lower_tensor(n.inputs_.get("w")))
    , b_(p.lower_tensor(n.inputs_.get("b"), 2))
    , y_(p.lower_tensor_batch(n.outputs_.get("y")))
  {
    chkCUDNN(cudnnCreateFilterDescriptor(&filter_desc_));
    chkCUDNN(cudnnCreateConvolutionDescriptor(&conv_desc_));


    chkCUDNN(cudnnSetFilter4dDescriptor(filter_desc_,
                                        x_->type_,
                                        p.tensor_format_,
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

  void exec(CudnnProgram &p) {
    float alpha = 1.0f, beta = 0.0f;

    chkCUDNN(cudnnConvolutionForward(ctx_->cudnn_, &alpha,
                                     x_->desc(),
                                     x_->deviceMem(),
                                     filter_desc_,
                                     w_->deviceMem(),
                                     conv_desc_,
                                     conv_fwd_algo_,
                                     p.workspace_, p.workspace_size_,
                                     &beta,
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

struct CudnnConvolutionBwd : public CudnnOperation {

  const std::shared_ptr<CudnnContext> ctx_;
  const std::shared_ptr<CudnnConvolutionFwd> fwd_;
  const std::shared_ptr<CudaTensor> dx_, dw_, db_, dy_;

  cudnnConvolutionBwdDataAlgo_t bwd_data_algo_;
  cudnnConvolutionBwdFilterAlgo_t bwd_filter_algo_;

  CudnnConvolutionBwd(CudnnProgram &p,
                      const Node &n,
                      std::shared_ptr<CudnnConvolutionFwd> fwd)
    : ctx_(p.ctx_)
    , fwd_(fwd)
    , dx_(fwd_->x_->grad_)
    , dw_(fwd_->w_->makeGrad())
    , db_(fwd_->b_ ? fwd_->b_->makeGrad() : nullptr)
    , dy_(fwd_->y_->makeGrad())
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
    printf("Convolution Bwd\n");
    if(dx_)
      printf("\tdx: %s\n", dx_->info().c_str());
    printf("\tdw: %s\n", dw_->info().c_str());
    if(db_)
      printf("\tdb: %s\n", db_->info().c_str());
    printf("\tdy: %s\n", dy_->info().c_str());
  }

  void exec(CudnnProgram &p) {

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
                                            &beta,
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
conv_make(CudnnProgram &p, const Node &n)
{
  auto f = std::make_shared<CudnnConvolutionFwd>(p, n);
  p.fwd(f);
  if(p.type_ == ProgramType::INFERENCE)
    return;
  auto b = std::make_shared<CudnnConvolutionBwd>(p, n, f);
  p.bwd(b);

  p.upd(std::make_shared<CudnnAdam>(p, f->w_, b->dw_));
  if(f->b_)
    p.upd(std::make_shared<CudnnAdam>(p, f->b_, b->db_));
}


//------------------------------------------------------------------------

struct CudnnBatchNormInference : public CudnnOperation {

  const std::shared_ptr<CudnnContext> ctx_;
  const std::shared_ptr<CudaTensor> x_, s_, b_, m_, v_, y_;
  const float epsilon_;

  CudnnBatchNormInference(CudnnProgram &p, const Node &n)
    : ctx_(p.ctx_)
    , x_(p.lower_tensor_batch(n.inputs_.get("x")))
    , s_(p.lower_tensor(n.inputs_.get("s"), 2))
    , b_(p.lower_tensor(n.inputs_.get("b"), 2))
    , m_(p.lower_tensor(n.inputs_.get("m"), 2))
    , v_(p.lower_tensor(n.inputs_.get("v"), 2))
    , y_(p.lower_tensor_batch(n.outputs_.get("y")))
    , epsilon_(n.attributes_.get("epsilon", 1e-5f))
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

  void exec(CudnnProgram &p) {
    float alpha = 1.0f, beta = 0.0f;
    chkCUDNN(cudnnBatchNormalizationForwardInference(ctx_->cudnn_,
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
                                                     epsilon_));

  }
};


struct CudnnBatchNormTrain : public CudnnBatchNormInference {

  const std::shared_ptr<CudaTensor> sm_, sv_;
  const float expavgf_;

  CudnnBatchNormTrain(CudnnProgram &p, const Node &n)
    : CudnnBatchNormInference(p, n)
    , sm_(std::make_shared<CudaTensor>(*m_, p.tensor_format_))
    , sv_(std::make_shared<CudaTensor>(*v_, p.tensor_format_))
    , expavgf_(n.attributes_.get("expavg", 0.1f))
  {}

  void print() const {
    CudnnBatchNormInference::print();
    printf("\tsm: %s\n", sm_->info().c_str());
    printf("\tsv: %s\n", sv_->info().c_str());
  }

  void exec(CudnnProgram &p) {
    float alpha = 1.0f, beta = 0.0f;
    chkCUDNN(cudnnBatchNormalizationForwardTraining(ctx_->cudnn_,
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
                                                    sv_->deviceMem()));


  }
};

struct CudnnBatchNormBwd : public CudnnOperation {

  const std::shared_ptr<CudnnContext> ctx_;
  const std::shared_ptr<CudaTensor> x_, dy_, dx_, s_, ds_, db_, sm_, sv_;
  const float epsilon_;

  CudnnBatchNormBwd(CudnnProgram &p, const Node &n,
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
  {}

  void print() const {
    printf("BatchNorm Bwd\n");
  }

  void exec(CudnnProgram &p) {
    float alpha = 1.0f, beta = 0.0f;

    chkCUDNN(cudnnBatchNormalizationBackward(ctx_->cudnn_,
                                             CUDNN_BATCHNORM_SPATIAL,
                                             &alpha, &beta,
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
batchnorm_make(CudnnProgram &p, const Node &n)
{
  p.infer(std::make_shared<CudnnBatchNormInference>(p, n));
  if(p.type_ == ProgramType::INFERENCE)
    return;

  auto f = std::make_shared<CudnnBatchNormTrain>(p, n);
  p.train(f);

  auto b = std::make_shared<CudnnBatchNormBwd>(p, n, *f);
  p.bwd(b);

  p.upd(std::make_shared<CudnnAdam>(p, f->s_, b->ds_));
  p.upd(std::make_shared<CudnnAdam>(p, f->b_, b->db_));
}


//------------------------------------------------------------------------

struct CudnnActivationFwd : public CudnnOperation {
  const std::shared_ptr<CudnnContext> ctx_;
  const std::shared_ptr<CudaTensor> x_, y_;
  cudnnActivationDescriptor_t desc_;

  CudnnActivationFwd(CudnnProgram &p, const Node &n,
                     cudnnActivationMode_t mode, float alpha)
    : ctx_(p.ctx_)
    , x_(p.lower_tensor_batch(n.inputs_.get("x")))
    , y_(p.lower_tensor_batch(n.outputs_.get("y")))
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

  void exec(CudnnProgram &p) {
    float alpha = 1.0f, beta = 0.0f;

    chkCUDNN(cudnnActivationForward(ctx_->cudnn_, desc_,
                                    &alpha,
                                    x_->desc(), x_->deviceMem(),
                                    &beta,
                                    y_->desc(), y_->deviceMem()));
  }
};


struct CudnnActivationBwd : public CudnnOperation {
  const std::shared_ptr<CudnnContext> ctx_;
  const std::shared_ptr<CudnnActivationFwd> fwd_;
  const std::shared_ptr<CudaTensor> dx_, dy_;
  cudnnActivationDescriptor_t desc_;

  CudnnActivationBwd(CudnnProgram &p, const Node &n,
                     const std::shared_ptr<CudnnActivationFwd> fwd)
    : ctx_(p.ctx_)
    , fwd_(fwd)
    , dx_(fwd->x_->makeGrad())
    , dy_(fwd->y_->makeGrad())
  {
  }

  ~CudnnActivationBwd()
  {
  }

  void print() const {
    printf("Activation Bwd\n");
    printf("\tdx: %s\n", dx_->info().c_str());
    printf("\tdy: %s\n", dy_->info().c_str());
  }

  void exec(CudnnProgram &p) {
    float alpha = 1.0f, beta = 0.0f;

    chkCUDNN(cudnnActivationBackward(ctx_->cudnn_, fwd_->desc_,
                                     &alpha,
                                     fwd_->y_->desc(), fwd_->y_->deviceMem(),
                                     dy_->desc(), dy_->deviceMem(),
                                     fwd_->x_->desc(), fwd_->x_->deviceMem(),
                                     &beta,
                                     dx_->desc(), dx_->deviceMem()));
  }
};



static void
relu_make(CudnnProgram &p, const Node &n)
{
  auto f = std::make_shared<CudnnActivationFwd>(p, n, CUDNN_ACTIVATION_RELU,
                                                0.0f);
  p.fwd(f);

  if(p.type_ == ProgramType::INFERENCE)
    return;
  p.bwd(std::make_shared<CudnnActivationBwd>(p, n, f));

}


//------------------------------------------------------------------------

struct CudnnPoolingFwd : public CudnnOperation {

  const std::shared_ptr<CudnnContext> ctx_;
  const std::shared_ptr<CudaTensor> x_, y_;
  cudnnPoolingDescriptor_t desc_;

  CudnnPoolingFwd(CudnnProgram &p, const Node &n, cudnnPoolingMode_t mode)
    : ctx_(p.ctx_)
    , x_(p.lower_tensor_batch(n.inputs_.get("x")))
    , y_(p.lower_tensor_batch(n.outputs_.get("y")))
  {
    const int size   = n.attributes_.get("size", 1);
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

  void exec(CudnnProgram &p) {
    float alpha = 1.0f, beta = 0.0f;

    chkCUDNN(cudnnPoolingForward(ctx_->cudnn_, desc_,
                                 &alpha,
                                 x_->desc(), x_->deviceMem(),
                                 &beta,
                                 y_->desc(), y_->deviceMem()));
  }
};




struct CudnnPoolingBwd : public CudnnOperation {

  const std::shared_ptr<CudnnContext> ctx_;
  const std::shared_ptr<CudnnPoolingFwd> fwd_;
  const std::shared_ptr<CudaTensor> dx_, dy_;

  CudnnPoolingBwd(CudnnProgram &p, const Node &n,
                  std::shared_ptr<CudnnPoolingFwd> fwd)
    : ctx_(p.ctx_)
    , fwd_(fwd)
    , dx_(fwd->x_->makeGrad())
    , dy_(fwd->y_->makeGrad())
  {
  }

  void print() const {
    printf("Pooling Bwd\n");
    printf("\tdx: %s\n", dx_->info().c_str());
    printf("\tdy: %s\n", dy_->info().c_str());
  }

  void exec(CudnnProgram &p) {
    float alpha = 1.0f, beta = 0.0f;

    chkCUDNN(cudnnPoolingBackward(ctx_->cudnn_, fwd_->desc_,
                                  &alpha,
                                  fwd_->y_->desc(), fwd_->y_->deviceMem(),
                                  dy_->desc(), dy_->deviceMem(),
                                  fwd_->x_->desc(), fwd_->x_->deviceMem(),
                                  &beta,
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
pooling_make(CudnnProgram &p, const Node &n, cudnnPoolingMode_t mode)
{
  auto f = std::make_shared<CudnnPoolingFwd>(p, n, mode);
  p.fwd(f);
  if(p.type_ == ProgramType::INFERENCE)
    return;
  p.bwd(std::make_shared<CudnnPoolingBwd>(p, n, f));
}


static void
maxpool_make(CudnnProgram &p, const Node &n)
{
  pooling_make(p, n, CUDNN_POOLING_MAX);
}

static void
avgpool_make(CudnnProgram &p, const Node &n)
{
  pooling_make(p, n, CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING);
}


//------------------------------------------------------------------------

struct CudnnSumFwd : public CudnnOperation {

  const std::shared_ptr<CudnnContext> ctx_;
  const std::shared_ptr<CudaTensor> x0_, x1_, y_;
  cudnnOpTensorDescriptor_t desc_;

  CudnnSumFwd(CudnnProgram &p, const Node &n)
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

  void exec(CudnnProgram &p) {

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
sum_make(CudnnProgram &p, const Node &n)
{
  auto f = std::make_shared<CudnnSumFwd>(p, n);
  p.fwd(f);
}

//------------------------------------------------------------------------


static std::shared_ptr<Node>
sum_transform(CudnnProgram &p, const Node &n)
{
  auto y = p.lower_tensor_batch(n.outputs_.get("y"));
  auto xvec = n.inputs_.getv("x");

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
                                        y->namePostfix("sum"));

  int64_t offset = 0;
  for(const auto &xh : xvec) {
    p.tensors_[xh] = std::make_shared<CudaTensor>(t->storage_,
                                                  xh->dims_,
                                                  offset,
                                                  (const int *)strides + 1,
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
                                         y->namePostfix("sum"));

  p.tensors_[ya] = ya;
  nn->inputs_["x"] = t;
  nn->outputs_["y"] = ya;
  return nn;
}


//------------------------------------------------------------------------

struct CudnnReduce : public CudnnOperation {

  const std::shared_ptr<CudnnContext> ctx_;
  const std::shared_ptr<CudaTensor> x_, y_;
  cudnnReduceTensorDescriptor_t desc_;

  CudnnReduce(CudnnProgram &p,
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

  void exec(CudnnProgram &p) {

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
cudnn_reduce_add_make(CudnnProgram &p, const Node &n)
{
  auto f = std::make_shared<CudnnReduce>(p,
                                         p.lower_tensor(n.inputs_.get("x")),
                                         p.lower_tensor(n.outputs_.get("y")),
                                         CUDNN_REDUCE_TENSOR_ADD);
  p.fwd(f);
}

//------------------------------------------------------------------------

struct CudnnGemmFwd : public CudnnOperation {

  const std::shared_ptr<CudnnContext> ctx_;
  const std::shared_ptr<CudaTensor> x_, w_, b_, y_;
  const int n_;
  const int num_inputs_;
  const int num_outputs_;
  const int transW_;

  CudnnGemmFwd(CudnnProgram &p, const Node &n)
    : ctx_(p.ctx_)
    , x_(p.lower_tensor_batch(n.inputs_.get("x")))
    , w_(p.lower_tensor(n.inputs_.get("w")))
    , b_(p.lower_tensor(n.inputs_.get("b"), 2))
    , y_(p.lower_tensor_batch(n.outputs_.get("y")))
    , n_(x_->dims_[0])
    , num_inputs_(x_->dims_[1])
    , num_outputs_(y_->dims_[1])
    , transW_(n.attributes_.get("transW", 0))
  {
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

  void exec(CudnnProgram &p) {

    float alpha = 1.0f, beta = 0.0f;
    __half halpha = 1.0f, hbeta = 0.0f;
    cublasOperation_t transA = CUBLAS_OP_T;
    cublasOperation_t transB = CUBLAS_OP_N;
    switch(x_->type_) {
    case CUDNN_DATA_FLOAT:
      chkCuda(cublasSgemm(ctx_->cublas_, transA, transB,
                          num_outputs_, n_, num_inputs_,
                          &alpha,
                          (const float *)w_->deviceMem(), num_inputs_,
                          (const float *)x_->deviceMem(), num_inputs_,
                          &beta,
                          (float *)y_->deviceMem(), num_outputs_));
      break;
    case CUDNN_DATA_HALF:
      chkCuda(cublasHgemm(ctx_->cublas_, transA, transB,
                          num_outputs_, n_, num_inputs_,
                          &halpha,
                          (const __half *)w_->deviceMem(), num_inputs_,
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


struct CudnnGemmBwd : public CudnnOperation {

  const std::shared_ptr<CudnnContext> ctx_;
  const std::shared_ptr<CudaTensor> dx_, dw_, db_, dy_, x_, w_;
  const int n_;
  const int num_inputs_;
  const int num_outputs_;
  const std::shared_ptr<CudaTensor> ones_;

  CudnnGemmBwd(CudnnProgram &p, const Node &n,
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
  {
  }

  void print() const {
    printf("Gemm Bwd\n");
    if(dx_)
      printf("\tdx: %s\n", dx_->info().c_str());
    printf("\tdw: %s\n", dw_->info().c_str());
    printf("\tdb: %s\n", db_->info().c_str());
    printf("\tdy: %s\n", dy_->info().c_str());
  }

  void exec(CudnnProgram &p) {
    float alpha = 1.0f, beta = 0.0f;
    //    __half halpha = 1.0f, hbeta = 0.0f;

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
                            &beta,
                            (float *)dx_->deviceMem(), num_inputs_));
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


      break;
#if 0
    case CUDNN_DATA_HALF:
      chkCuda(cublasHgemm(ctx_->cublas_, CUBLAS_OP_N, CUBLAS_OP_T,
                          num_inputs_, num_outputs_, n_,
                          &halpha,
                          (const __half *)input_->deviceMem(), num_inputs_,
                          (const __half *)output_grad_->deviceMem(),
                          num_outputs_,
                          &hbeta,
                          (__half *)weights_grad_->deviceMem(), num_inputs_));

      // No cublasSgemv() for half type, so do matrix*matrix instead
      chkCuda(cublasHgemm(ctx_->cublas_, CUBLAS_OP_N, CUBLAS_OP_T,
                          1, num_outputs_, input_->n,
                          &halpha,
                          (const __half *)batch_of_one_.deviceMem(),
                          1,
                          (const __half *)output_grad_->deviceMem(),
                          num_outputs_,
                          &hbeta,
                          (__half *)bias_grad_->deviceMem(), 1));

      if(input_grad_ != NULL) {
        chkCuda(cublasHgemm(ctx_->cublas_, CUBLAS_OP_N, CUBLAS_OP_N,
                            num_inputs_, input_->n, num_outputs_,
                            &halpha,
                            (const __half *)weights_->deviceMem(), num_inputs_,
                            (const __half *)output_grad_->deviceMem(),
                            num_outputs_,
                            &hbeta,
                            (__half *)input_grad_->deviceMem(), num_inputs_));
      }
      break;
#endif
    default:
      abort();
    }
  }
};

static void
fc_make(CudnnProgram &p, const Node &n)
{
  auto f = std::make_shared<CudnnGemmFwd>(p, n);
  p.fwd(f);

  if(p.type_ == ProgramType::INFERENCE)
    return;
  auto b = std::make_shared<CudnnGemmBwd>(p, n, f);
  p.bwd(b);

  p.upd(std::make_shared<CudnnAdam>(p, f->w_, b->dw_));
  if(f->b_)
    p.upd(std::make_shared<CudnnAdam>(p, f->b_, b->db_));
}


//------------------------------------------------------------------------

struct CudnnSoftmaxFwd : public CudnnOperation {

  const std::shared_ptr<CudnnContext> ctx_;
  const std::shared_ptr<CudaTensor> x_, y_;

  CudnnSoftmaxFwd(CudnnProgram &p, const Node &n)
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

  void exec(CudnnProgram &p) {

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
softmax_make(CudnnProgram &p, const Node &n)
{
  auto f = std::make_shared<CudnnSoftmaxFwd>(p, n);
  p.fwd(f);
}

//------------------------------------------------------------------------
struct CudnnCatClassifierFwd : public CudnnOperation {

  const std::shared_ptr<CudnnContext> ctx_;
  const std::shared_ptr<CudaTensor> x_, p_, y_;

  CudnnCatClassifierFwd(CudnnProgram &p, const Node &n)
    : ctx_(p.ctx_)
    , x_(p.lower_tensor_batch(n.inputs_.get("x")))
    , p_(std::make_shared<CudaTensor>(x_->data_type_, x_->dims_,
                                      p.tensor_format_))
    , y_(p.lower_tensor_batch(n.outputs_.get("y")))
  {
  }

  void print() const {
    printf("CatClassifier Fwd\n");
    printf("\tx: %s\n", x_->info().c_str());
    printf("\tp: %s\n", p_->info().c_str());
    printf("\ty: %s\n", y_->info().c_str());
  }

  void exec(CudnnProgram &p) {

    float alpha = 1.0f, beta = 0.0f;
    chkCUDNN(cudnnSoftmaxForward(ctx_->cudnn_,
                                 CUDNN_SOFTMAX_ACCURATE,
                                 CUDNN_SOFTMAX_MODE_CHANNEL,
                                 &alpha,
                                 x_->desc(), x_->deviceMem(),
                                 &beta,
                                 p_->desc(), p_->deviceMem()));

    switch(p_->type_) {
    case CUDNN_DATA_FLOAT:
      catclassifier_pred_float_i32(p_->dims_[0],
                                   (const float *)p_->deviceMem(),
                                   (int32_t *)y_->deviceMem(), p_->dims_[1]);
      break;
    default:
      abort();
    }
  }
};

struct CudnnCatClassifierBwd : public CudnnOperation {

  const std::shared_ptr<CudnnContext> ctx_;
  const std::shared_ptr<CudnnCatClassifierFwd> fwd_;
  const std::shared_ptr<CudaTensor> dx_, dy_, loss_;

  CudnnCatClassifierBwd(CudnnProgram &p, const Node &n,
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
    printf("\tdx: %s\n", dx_->info().c_str());
    printf("\tdy: %s\n", dy_->info().c_str());
  }

  void exec(CudnnProgram &p) {

    const int n = fwd_->p_->dims_[0];
    const int c = fwd_->p_->dims_[1];
    const float scale = 1.0f / n;

    switch(fwd_->p_->type_) {
    case CUDNN_DATA_FLOAT:
      catclassifier_backprop_float_i32(n,
                                       (const float *)fwd_->p_->deviceMem(),
                                       (float *)dx_->deviceMem(),
                                       (const int32_t *)dy_->deviceMem(),
                                       loss_ ? (float *)loss_->deviceMem() : NULL,
                                       c, scale);
      break;
    default:
      abort();
    }

    if(p.debug_) {
      fwd_->x_->print("catclassifier.x");
      fwd_->p_->print("catclassifier.p");
      dy_->print("catclassifier.dy");
      dx_->print("catclassifier.dx");
    }

  }

};

static void
catclassifier_make(CudnnProgram &p, const Node &n)
{
  auto f = std::make_shared<CudnnCatClassifierFwd>(p, n);
  p.fwd(f);
  if(p.type_ == ProgramType::INFERENCE)
    return;
  p.bwd(std::make_shared<CudnnCatClassifierBwd>(p, n, f));
}


//------------------------------------------------------------------------

struct CudnnTransform : public CudnnOperation {

  const std::shared_ptr<CudnnContext> ctx_;
  const std::shared_ptr<CudaTensor> x_, y_;

  CudnnTransform(CudnnProgram &p,
                 std::shared_ptr<CudaTensor> x,
                 std::shared_ptr<CudaTensor> y)
    : ctx_(p.ctx_)
    , x_(x)
    , y_(y)
  {}

  void print() const {
    printf("Transform\n");
    printf("\tx: %s\n", x_->info().c_str());
    printf("\ty: %s\n", y_->info().c_str());
  }

  void exec(CudnnProgram &p) {
    float alpha = 1.0f, beta = 0.0f;
    chkCUDNN(cudnnTransformTensor(ctx_->cudnn_,
                                  &alpha,
                                  x_->desc(),
                                  x_->deviceMem(),
                                  &beta,
                                  y_->desc(),
                                  y_->deviceMem()));
  }
};

static void
concat_make(CudnnProgram &p, const Node &n)
{
  auto y = p.lower_tensor_batch(n.outputs_.get("y"));
  auto element_offset = std::vector<int64_t>(y->dims_.size(), 0);
  const int axis = 1;
  for(const auto &xh : n.inputs_.getv("x")) {
    auto x = p.lower_tensor_batch(xh);
    auto y2 = std::make_shared<CudaTensor>(y, x->dims_, element_offset,
                                           y->namePostfix("concat.alias"));
    p.fwd(std::make_shared<CudnnTransform>(p, x, y2));
    element_offset[axis] += xh->dims_[axis];
  }
}


static void
concat_transform(CudnnProgram &p, const Node &n)
{
  const int axis = 1;

  auto y = p.lower_tensor_batch(n.outputs_.get("y"));
  auto dy = y->makeGrad();
  auto element_offset = std::vector<int64_t>(y->dims_.size(), 0);

  for(const auto &xh : n.inputs_.getv("x")) {
    std::vector<int64_t> dims = xh->dims_;
    dims[0] = p.batch_size_;

    auto x = std::make_shared<CudaTensor>(y, dims, element_offset,
                                          xh->namePostfix("alias"));
    p.tensors_[xh] = x;
    x->grad_ = std::make_shared<CudaTensor>(dy, dims, element_offset,
                                            xh->namePostfix("alias"));
    element_offset[axis] += xh->dims_[axis];
  }
}


//------------------------------------------------------------------------



struct CudnnConvert : public CudnnOperation {

  const std::shared_ptr<CudnnContext> ctx_;
  const std::shared_ptr<CudaTensor> x_, y_;
  const float scale_;
  void (*algo_)(const void *src, void *dst, int elements, float scale);

  CudnnConvert(CudnnProgram &p,
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
    } else {
      abort();
    }
  }

  void print() const {
    printf("Convert %zd elements\n", (size_t)x_->elements_);
    printf("\tx: %s\n", x_->info().c_str());
    printf("\ty: %s\n", y_->info().c_str());
  }

  void exec(CudnnProgram &p) {
    algo_(x_->deviceMem(), y_->deviceMem(), x_->elements_, scale_);
  }

};


static void
convert_make(CudnnProgram &p, const Node &n)
{
  auto x = p.lower_tensor_batch(n.inputs_.get("x"));
  auto y = p.lower_tensor_batch(n.outputs_.get("y"));
  auto scale = n.attributes_.get("scale", 1.0f);

  p.fwd(std::make_shared<CudnnConvert>(p, x, y, scale));

}

//------------------------------------------------------------------------

struct CudnnMathOp : public CudnnOperation {

  const std::shared_ptr<CudnnContext> ctx_;
  const std::shared_ptr<CudaTensor> a_, b_, c_;
  cudnnOpTensorDescriptor_t desc_;

  CudnnMathOp(CudnnProgram &p,
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

  void exec(CudnnProgram &p) {

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
add_make(CudnnProgram &p, const Node &n)
{
  auto f = std::make_shared<CudnnMathOp>(p,
                                         p.lower_tensor_batch(n.outputs_.get("x")),
                                         p.lower_tensor(n.outputs_.get("b")),
                                         p.lower_tensor_batch(n.outputs_.get("y")),
                                         CUDNN_OP_TENSOR_ADD);
  p.fwd(f);
}


static void
mul_make(CudnnProgram &p, const Node &n)
{
  auto f = std::make_shared<CudnnMathOp>(p,
                                         p.lower_tensor_batch(n.outputs_.get("x")),
                                         p.lower_tensor(n.outputs_.get("s")),
                                         p.lower_tensor_batch(n.outputs_.get("y")),
                                         CUDNN_OP_TENSOR_MUL);
  p.fwd(f);
}




//------------------------------------------------------------------------

static std::vector<std::shared_ptr<Node>>
reshape_transform(CudnnProgram &p, const Node &n)
{
  auto x = p.lower_tensor_batch(n.inputs_.get("x"), CUDNN_TENSOR_NCHW);
  auto dx = x->makeGrad();
  auto y = n.outputs_.get("y");

  Dims dims(y->dims_);
  dims[0] = p.batch_size_;

  auto yl = std::make_shared<CudaTensor>(x->storage_,
                                         dims, CUDNN_TENSOR_NCHW,
                                         x->namePostfix("reshape"));

  p.tensors_[y] = yl;
  yl->grad_ = std::make_shared<CudaTensor>(dx->storage_,
                                           dims, CUDNN_TENSOR_NCHW,
                                           x->namePostfix("reshape"));
  return {};
}

//------------------------------------------------------------------------

static std::vector<std::shared_ptr<Node>>
dropout_transform(CudnnProgram &p, const Node &n)
{
  auto x = p.lower_tensor_batch(n.inputs_.get("x"));
  auto y = n.outputs_.get("y");


  p.tensors_[y] = std::make_shared<CudaTensor>(x->storage_,
                                               x->dims_, p.tensor_format_,
                                               x->namePostfix("dropout"));
  return {};
}


//------------------------------------------------------------------------

struct CudnnPrintTensor : public CudnnOperation {

  const std::string prefix_;
  const std::shared_ptr<Tensor> x_;

  CudnnPrintTensor(const std::string &prefix, std::shared_ptr<Tensor> x)
    : prefix_(prefix)
    , x_(x)
  {}

  void exec(CudnnProgram &p) {
    if(!p.debug_)
      return;
    x_->print(prefix_.c_str(), 4);
  }
};

//------------------------------------------------------------------------

struct CudnnPrintStatsTensor : public CudnnOperation {

  const std::string prefix_;
  const std::shared_ptr<Tensor> x_;

  CudnnPrintStatsTensor(const std::string &prefix, std::shared_ptr<Tensor> x)
    : prefix_(prefix)
    , x_(x)
  {}

  void print() const {
  }

  void exec(CudnnProgram &p) {
    if(!p.debug_)
      return;
    printf("%s: %s\n", prefix_.c_str(), x_->info().c_str());
    x_->printStats(prefix_.c_str());
  }
};


//------------------------------------------------------------------------

struct CudnnHalt : public CudnnOperation {

  CudnnHalt()
  {
  }

  void exec(CudnnProgram &p) {
    fprintf(stderr, "Stopped by CudnnHalt\n");
    exit(1);
  }
};

//------------------------------------------------------------------------

static const struct {
  const char *name;
  void (*create_op)(CudnnProgram &p, const Node &n);
  std::vector<std::shared_ptr<Node>> (*transform_op)(CudnnProgram &p,
                                                     const Node &n);
} nodetypes[] = {
  { "add",           add_make },
  { "avgpool",       avgpool_make },
  { "batchnorm",     batchnorm_make },
  { "catclassifier", catclassifier_make },
  { "concat",        concat_make },
  { "conv",          conv_make },
  { "convert",       convert_make },
  { "cudnn.reduce.add",  cudnn_reduce_add_make },
  { "dropout",       NULL, dropout_transform },
  { "fc",            fc_make },
  { "maxpool",       maxpool_make },
  { "mul",           mul_make },
  { "relu",          relu_make },
  { "reshape",       NULL, reshape_transform },
  { "softmax",       softmax_make },
  { "sum",           sum_make }
};


static void
generate_operation(CudnnProgram &p, const Node &n)
{
  for(size_t i = 0; i < sizeof(nodetypes) / sizeof(nodetypes[0]); i++) {
    if(n.type_ == nodetypes[i].name) {
      nodetypes[i].create_op(p, n);
      return;
    }
  }
  printf("Cant emit operation for node type %s\n", n.type_.c_str());
  abort();
}



static std::vector<std::shared_ptr<Node>>
pass_remove_concat(CudnnProgram &p,
                   const std::vector<std::shared_ptr<Node>> &nodes)
{
  std::vector<std::shared_ptr<Node>> r;

  for(ssize_t i = nodes.size() - 1; i >= 0; i--) {
    auto &n = nodes[i];
    if(n->type_ == "concat") {
      concat_transform(p, *n);
#if 0
    } else if(n->type_ == "sum") {
      r.insert(r.begin(), sum_transform(p, *n));
#endif
    } else {
      r.insert(r.begin(), n);
    }
  }
  return r;
}


static std::vector<std::shared_ptr<Node>>
pass_node_transform(CudnnProgram &p,
                    const std::vector<std::shared_ptr<Node>> &nodes)
{
  std::vector<std::shared_ptr<Node>> r;

  for(const auto &n : nodes) {
    std::vector<std::shared_ptr<Node>> v{n};
    for(size_t i = 0; i < sizeof(nodetypes) / sizeof(nodetypes[0]); i++) {
      if(n->type_ == nodetypes[i].name && nodetypes[i].transform_op) {
        v = nodetypes[i].transform_op(p, *n);
        break;
      }
    }
    r.insert(r.end(), v.begin(), v.end());
  }
  return r;
}



std::shared_ptr<Program>
CudnnContext::createProgram(const Graph &g,
                            ProgramType type,
                            int batch_size,
                            float learning_rate,
                            TensorLayout layout)
{

  cudnnTensorFormat_t tensor_format = CUDNN_TENSOR_NHWC;
  if(layout == TensorLayout::NCHW)
    tensor_format = CUDNN_TENSOR_NCHW;

  auto p = std::make_shared<CudnnProgram>(shared_from_this(),
                                          tensor_format,
                                          type, batch_size,
                                          learning_rate);

  auto nodes = pass_remove_concat(*p, g.nodes_);

  nodes = pass_node_transform(*p, nodes);

  for(const auto &n : nodes) {
    generate_operation(*p, *n);
  }

  return p;
}





}
