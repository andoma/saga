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
                                         int batch_size);

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
               int batch_size)
    : ctx_(ctx)
    , tensor_format_(tensor_format)
    , type_(type)
    , batch_size_(batch_size)
    , workspace_(NULL)
    , workspace_size_(0)
    , workspace_requested_(0)
  {
  }

  ~CudnnProgram()
  {
    chkCuda(cudaFree(workspace_));
  }


  void exec();

  void print() const;

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

  std::unordered_map<std::shared_ptr<Tensor>,
                     std::shared_ptr<CudaTensor>> tensors_;

  std::vector<std::shared_ptr<CudnnOperation>> fwd_operations_;
  std::vector<std::shared_ptr<CudnnOperation>> bwd_operations_;
  std::vector<std::shared_ptr<CudnnOperation>> upd_operations_;

  void *workspace_;
  size_t workspace_size_;
  size_t workspace_requested_;

  std::shared_ptr<CudaTensor> lower_tensor(std::shared_ptr<Tensor> src,
                                           size_t dimensions = 0);

  std::shared_ptr<CudaTensor> lower_tensor_batch(std::shared_ptr<Tensor> src);

  std::shared_ptr<CudaTensor> grad(std::shared_ptr<CudaTensor> src) {
    if(!src)
      return nullptr;
    return std::make_shared<CudaTensor>(*src, tensor_format_, "grad");
  }

  void fwd(const std::shared_ptr<CudnnOperation> &op)
  {
    fwd_operations_.push_back(op);
  }

  void bwd(const std::shared_ptr<CudnnOperation> &op)
  {
    bwd_operations_.insert(bwd_operations_.begin(), op);
  }
};


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
CudnnProgram::lower_tensor_batch(std::shared_ptr<Tensor> src)
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
                                        dims, tensor_format_,
                                        src->name_);

  t->copyFrom(*src);
  tensors_[src] = t;
  return t;
}






class CudnnOperation {
public:
  virtual ~CudnnOperation() {}
  virtual void exec(CudnnProgram &p) = 0;
  virtual void print() const = 0;
};

//------------------------------------------------------------------------


void
CudnnProgram::exec()
{
  allocWorkspace();
  for(const auto &op : fwd_operations_) {
    op->exec(*this);
  }
  for(const auto &op : bwd_operations_) {
    op->exec(*this);
  }
  for(const auto &op : upd_operations_) {
    op->exec(*this);
  }
}

void
CudnnProgram::print() const
{
  for(const auto &t : inputs_) {
    printf("Input: %s\n", t->info().c_str());
  }

  printf("Forward:\n");
  for(const auto &op : fwd_operations_) {
    op->print();
  }

  printf("Backward:\n");
  for(const auto &op : bwd_operations_) {
    op->print();
  }


  for(const auto &t : outputs_) {
    printf("Output: %s\n", t->info().c_str());
  }
}


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
    , dx_(p.lower_tensor_batch(n.outputs_.get("dx")))
    , dw_(p.grad(fwd->w_))
    , db_(p.grad(fwd->b_))
    , dy_(p.lower_tensor_batch(n.inputs_.get("dy")))
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
  }
};


static void
conv_make(CudnnProgram &p, const Node &n)
{
  auto f = std::make_shared<CudnnConvolutionFwd>(p, n);
  p.fwd(f);
  if(p.type_ == ProgramType::INFERENCE)
    return;
  p.bwd(std::make_shared<CudnnConvolutionBwd>(p, n, f));

}


//------------------------------------------------------------------------

struct CudnnBatchNormFwd : public CudnnOperation {

  const std::shared_ptr<CudnnContext> ctx_;
  const std::shared_ptr<CudaTensor> x_, s_, b_, m_, v_, y_;
  const float epsilon_;

  CudnnBatchNormFwd(CudnnProgram &p, const Node &n)
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
    printf("\ts: %s\n", x_->info().c_str());
    printf("\tb: %s\n", x_->info().c_str());
    printf("\tm: %s\n", x_->info().c_str());
    printf("\tv: %s\n", x_->info().c_str());
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

static void
batchnorm_make(CudnnProgram &p, const Node &n)
{
  auto f = std::make_shared<CudnnBatchNormFwd>(p, n);
  p.fwd(f);
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
    , dx_(p.lower_tensor_batch(n.outputs_.get("dx")))
    , dy_(p.lower_tensor_batch(n.inputs_.get("dy")))
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
                                     dx_->desc(),
                                     dx_->deviceMem()));
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
    , dx_(p.lower_tensor_batch(n.outputs_.get("dx")))
    , dy_(p.lower_tensor_batch(n.inputs_.get("dy")))
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

struct CudnnGemmFwd : public CudnnOperation {

  const std::shared_ptr<CudnnContext> ctx_;
  const std::shared_ptr<CudaTensor> x_, w_, b_, y_;
  const int n_;
  const int num_inputs_;
  const int num_outputs_;
  const int transW_;
  //  cudnnOpTensorDescriptor_t desc_;

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
    printf("Gemm Fwd\n");
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
  }
};


struct CudnnGemmBwd : public CudnnOperation {

  const std::shared_ptr<CudnnContext> ctx_;
  const std::shared_ptr<CudaTensor> dx_, dw_, db_, dy_, x_;
  const int n_;
  const int num_inputs_;
  const int num_outputs_;
  const std::shared_ptr<CudaTensor> ones_;

  CudnnGemmBwd(CudnnProgram &p, const Node &n,
               std::shared_ptr<CudnnGemmFwd> fwd)
    : ctx_(p.ctx_)
    , dx_(p.lower_tensor_batch(n.outputs_.get("dx")))
    , dw_(p.grad(fwd->w_))
    , db_(p.grad(fwd->b_))
    , dy_(p.lower_tensor_batch(n.inputs_.get("dy")))
    , x_(fwd->x_)
    , n_(x_->dims_[0])
    , num_inputs_(x_->dims_[1])
    , num_outputs_(fwd->y_->dims_[1])
    , ones_(p.lower_tensor(Tensor::make(x_->data_type_, {n_,1}, 1, 0)))
  {
  }

  void print() const {
    printf("Gemm Bwd\n");
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
                            (const float *)dw_->deviceMem(), num_inputs_,
                            (const float *)dy_->deviceMem(),
                            num_outputs_,
                            &beta,
                            (float *)dx_->deviceMem(), num_inputs_));
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
  p.bwd(std::make_shared<CudnnGemmBwd>(p, n, f));

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
    , dx_(p.lower_tensor_batch(n.outputs_.get("dx")))
    , dy_(p.lower_tensor_batch(n.inputs_.get("dy")))
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
                                       (float *)loss_->deviceMem(),
                                       c, scale);
      break;
    default:
      abort();
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

static void
reshape_make(CudnnProgram &p, const Node &n)
{
  auto x = p.lower_tensor_batch(n.inputs_.get("x"));
  auto y = n.outputs_.get("y");

  Dims dims(y->dims_);
  dims[0] = p.batch_size_;

  p.tensors_[y] = std::make_shared<CudaTensor>(x->storage_,
                                               dims, p.tensor_format_,
                                               x->namePostfix("reshape"));
}


static void
dropout_make(CudnnProgram &p, const Node &n)
{
  auto x = p.lower_tensor_batch(n.inputs_.get("x"));
  auto y = n.outputs_.get("y");

  p.tensors_[y] = std::make_shared<CudaTensor>(x->storage_,
                                               x->dims_, p.tensor_format_,
                                               x->namePostfix("dropout"));
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

  void exec(CudnnProgram &p) {
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

#define DEFNODE(x) { #x, x##_make }

static const struct {
  const char *name;
  void (*create_op)(CudnnProgram &p, const Node &n);
} nodetypes[] = {
  DEFNODE(add),
  DEFNODE(avgpool),
  DEFNODE(batchnorm),
  DEFNODE(catclassifier),
  DEFNODE(concat),
  DEFNODE(conv),
  DEFNODE(dropout),
  DEFNODE(fc),
  DEFNODE(maxpool),
  DEFNODE(mul),
  DEFNODE(relu),
  DEFNODE(reshape),
  DEFNODE(softmax),
  DEFNODE(sum),
};

#undef DEFNODE

//------------------------------------------------------------------------

static void
generate_operation(CudnnProgram &p, const Node &n)
{
  for(size_t i = 0; i < sizeof(nodetypes) / sizeof(nodetypes[0]); i++) {
    if(n.type_ == nodetypes[i].name) {
      nodetypes[i].create_op(p, n);
      return;
    }
  }
  printf("Cant emit forward operation for node type %s\n", n.type_.c_str());
  abort();
}


std::shared_ptr<Program>
CudnnContext::createProgram(const Graph &g,
                            ProgramType type,
                            int batch_size)
{
  auto p = std::make_shared<CudnnProgram>(shared_from_this(),
                                          CUDNN_TENSOR_NCHW,
                                          type, batch_size);

  for(const auto &n : g.inputs_) {
    p->inputs_.insert(p->lower_tensor_batch(n));
  }

  for(const auto &n : g.nodes_) {
    generate_operation(*p, *n);
  }

  for(const auto &n : g.outputs_) {
    p->outputs_.insert(p->lower_tensor_batch(n));
  }
  return p;
}





}
