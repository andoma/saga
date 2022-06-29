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
#include "saga.hpp"
#include "tensor.hpp"
#include "context.hpp"

#include "cuda_common.hpp"
#include "cuda_tensor.hpp"
#include "cuda_kernels.hpp"

namespace saga {

struct CudnnOperation : public CudaOperation {
    CudnnOperation(const std::string &name) : CudaOperation(name) {}

    virtual cudnnStatus_t exec(CudaProgram &p) = 0;

    const char *exec(CudaProgram &p, long batch)
    {
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

    CudnnAdam(CudaProgram &p, std::shared_ptr<CudaTensor> weights,
              std::shared_ptr<CudaTensor> gradient)
      : CudaOperation("adam")
      , weights_(weights)
      , gradient_(gradient)
      , learning_rate_(p.m_learning_rate)
      , iter_(0)
      , ctx_(p.m_ctx)
      , m_elements(weights->dims_.elements())
    {
        assert(weights->dims_ == gradient->dims_);

        switch(weights->data_type_) {
        case Tensor::DataType::FLOAT:
            // Allocate 2x floats for each weight (m and v)
            chkCuda(cudaMallocManaged(&temp_, m_elements * 2 * sizeof(float)));
            chkCuda(cudaMemset(temp_, 0, m_elements * 2 * sizeof(float)));
            break;

        case Tensor::DataType::HALF:
            // Allocate 3x floats for each weight (m and v and float32 copy)
            chkCuda(cudaMallocManaged(&temp_, m_elements * 3 * sizeof(float),
                                      cudaMemAttachGlobal));
            {
                const __half *src = (const __half *)weights->deviceMem();
                float *dst = temp_;
                for(size_t i = 0; i < m_elements; i++) {
                    *dst++ = 0;
                    *dst++ = 0;
                    *dst++ = *src++;
                }
            }
            break;

        default:
            break;
        }
    }

    ~CudnnAdam() { chkCuda(cudaFree(temp_)); }

    std::vector<std::shared_ptr<CudaTensor>> getInputs() const override
    {
        return {gradient_, weights_};
    }

    std::vector<std::shared_ptr<CudaTensor>> getOutputs() const override
    {
        return {weights_};
    }

    const char *exec(CudaProgram &p, long batch) override
    {
        const int i = ++iter_;

#define ADAM_B1 0.9
#define ADAM_B2 0.999

        const float b1t = 1.0 / (1.0 - pow(ADAM_B1, i));
        const float b2t = 1.0 / (1.0 - pow(ADAM_B2, i));

        switch(weights_->data_type_) {
        case Tensor::DataType::FLOAT:
            adam_float(m_elements, (float *)weights_->deviceMem(),
                       (const float *)gradient_->deviceMem(), (float *)temp_,
                       b1t, b2t, learning_rate_, p.m_ctx->m_stream);
            break;
        case Tensor::DataType::HALF:
            p.m_mp_enabled = true;
            adam_mixed(m_elements, 1.0f / p.m_mp_scaling,
                       (__half *)weights_->deviceMem(),
                       (const __half *)gradient_->deviceMem(), (float *)temp_,
                       b1t, b2t, learning_rate_, (int *)p.m_check_result,
                       p.m_ctx->m_stream);
            break;
        default:
            return "Unsupported tensor datatype";
        }
        return NULL;
    }
    const std::shared_ptr<CudaContext> ctx_;
    const size_t m_elements;
};

//------------------------------------------------------------------------

struct CudnnAddTensor : public CudnnOperation {
    const std::shared_ptr<CudaTensor> x_, y_;

    CudnnAddTensor(std::shared_ptr<CudaTensor> x, std::shared_ptr<CudaTensor> y)
      : CudnnOperation("add"), x_(x), y_(y)
    {
    }

    cudnnStatus_t exec(CudaProgram &p)
    {
        float alpha = 1.0f;
        return cudnnAddTensor(p.m_ctx->m_cudnn, &alpha, x_->desc(),
                              x_->deviceMem(), &alpha, y_->desc(),
                              y_->deviceMem());
    }

    std::vector<std::shared_ptr<CudaTensor>> getInputs() const override
    {
        return {x_, y_};
    }

    std::vector<std::shared_ptr<CudaTensor>> getOutputs() const override
    {
        return {y_};
    }
};

//------------------------------------------------------------------------

struct CudnnTransform : public CudnnOperation {
    const std::shared_ptr<CudaTensor> a_, b_;
    const float beta_;

    CudnnTransform(std::shared_ptr<CudaTensor> a, std::shared_ptr<CudaTensor> b,
                   float beta)
      : CudnnOperation("transform"), a_(a), b_(b), beta_(beta)
    {
    }

    cudnnStatus_t exec(CudaProgram &p)
    {
        float alpha = 1.0f;
        return cudnnTransformTensor(p.m_ctx->m_cudnn, &alpha, a_->desc(),
                                    a_->deviceMem(), &beta_, b_->desc(),
                                    b_->deviceMem());
    }

    std::vector<std::shared_ptr<CudaTensor>> getInputs() const override
    {
        auto r = std::vector{a_};
        if(beta_)
            r.push_back(b_);
        return r;
    }

    std::vector<std::shared_ptr<CudaTensor>> getOutputs() const override
    {
        return {b_};
    }

    std::string info() const override
    {
        if(beta_) {
            return "beta=" + std::to_string(beta_);
        }
        return "";
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
        return "Gemm";
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

static std::string
cudnn_conv_hash_key(CudaTensor &x, CudaTensor &w, CudaTensor &y, int pad,
                    int stride)
{
    auto xkey = x.hashkey();
    auto wkey = w.hashkey();
    auto ykey = y.hashkey();

    char buf[512];
    snprintf(buf, sizeof(buf), "%s|%s|%s|%d|%d", xkey.c_str(), wkey.c_str(),
             ykey.c_str(), pad, stride);
    return buf;
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

    const char *setup(CudaProgram &p, CudaTensor &x, CudaTensor &w,
                      CudaTensor &y, int pad, int stride, bool bwd)
    {
        auto algo_key = cudnn_conv_hash_key(x, w, y, pad, stride);

        cudnnStatus_t s;
        s = cudnnSetFilter4dDescriptor(filter_desc_, x.m_type,
                                       p.tensorFormat(x), w.dims_[0],
                                       w.dims_[1], w.dims_[2], w.dims_[3]);
        if(s)
            return cudnnGetErrorString(s);

        s = cudnnSetConvolutionMathType(conv_desc_, CUDNN_TENSOR_OP_MATH);
        if(s)
            return cudnnGetErrorString(s);

        s = cudnnSetConvolution2dDescriptor(
            conv_desc_, pad, pad, stride, stride, 1, 1, CUDNN_CROSS_CORRELATION,
            CUDNN_DATA_FLOAT);
        if(s)
            return cudnnGetErrorString(s);

        auto fwd_algo_key = std::string("conv.fwd;") + algo_key;
        if(auto it = p.m_algo_hash.find(fwd_algo_key);
           it != p.m_algo_hash.end()) {
            conv_fwd_algo_ = (cudnnConvolutionFwdAlgo_t)it->second;
        } else {
            int count;
            s = cudnnGetConvolutionForwardAlgorithmMaxCount(p.m_ctx->m_cudnn,
                                                            &count);
            if(s)
                return cudnnGetErrorString(s);

            cudnnConvolutionFwdAlgoPerf_t fwdalgos[count];

            s = cudnnFindConvolutionForwardAlgorithm(
                p.m_ctx->m_cudnn, x.m_desc, filter_desc_, conv_desc_, y.m_desc,
                count, &count, fwdalgos);
            if(s)
                return cudnnGetErrorString(s);

            if(count == 0)
                return "No forwarding algo found";

            conv_fwd_algo_ = fwdalgos[0].algo;
            p.m_algo_hash[fwd_algo_key] = conv_fwd_algo_;
        }
        size_t workspace;
        s = cudnnGetConvolutionForwardWorkspaceSize(
            p.m_ctx->m_cudnn, x.m_desc, filter_desc_, conv_desc_, y.m_desc,
            conv_fwd_algo_, &workspace);
        if(s)
            return cudnnGetErrorString(s);

        p.m_ctx->m_workspace.request(workspace);

        if(!bwd)
            return NULL;

        auto bwd_data_algo_key = std::string("conv.bwd.data;") + algo_key;
        if(auto it = p.m_algo_hash.find(bwd_data_algo_key);
           it != p.m_algo_hash.end()) {
            bwd_data_algo_ = (cudnnConvolutionBwdDataAlgo_t)it->second;
        } else {
            int count;
            s = cudnnGetConvolutionBackwardDataAlgorithmMaxCount(
                p.m_ctx->m_cudnn, &count);
            if(s)
                return cudnnGetErrorString(s);

            cudnnConvolutionBwdDataAlgoPerf_t bwdalgos[count];

            s = cudnnFindConvolutionBackwardDataAlgorithm(
                p.m_ctx->m_cudnn, filter_desc_, y.desc(), conv_desc_, x.desc(),
                count, &count, bwdalgos);
            if(s)
                return cudnnGetErrorString(s);

            if(count == 0)
                return "No backward data algo found";

            bwd_data_algo_ = bwdalgos[0].algo;
            p.m_algo_hash[bwd_data_algo_key] = bwd_data_algo_;
        }
        s = cudnnGetConvolutionBackwardDataWorkspaceSize(
            p.m_ctx->m_cudnn, filter_desc_, y.desc(), conv_desc_, x.desc(),
            bwd_data_algo_, &workspace);
        if(s)
            return cudnnGetErrorString(s);

        p.m_ctx->m_workspace.request(workspace);

        auto bwd_filter_algo_key = std::string("conv.bwd.filter;") + algo_key;
        if(auto it = p.m_algo_hash.find(bwd_filter_algo_key);
           it != p.m_algo_hash.end()) {
            bwd_filter_algo_ = (cudnnConvolutionBwdFilterAlgo_t)it->second;
        } else {
            int count;
            s = cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(
                p.m_ctx->m_cudnn, &count);
            if(s)
                return cudnnGetErrorString(s);

            cudnnConvolutionBwdFilterAlgoPerf_t filteralgos[count];

            s = cudnnFindConvolutionBackwardFilterAlgorithm(
                p.m_ctx->m_cudnn, x.desc(), y.desc(), conv_desc_, filter_desc_,
                count, &count, filteralgos);
            if(s)
                return cudnnGetErrorString(s);

            if(count == 0)
                return "No backward filter algo found";

            bwd_filter_algo_ = filteralgos[0].algo;
            p.m_algo_hash[bwd_filter_algo_key] = bwd_filter_algo_;
        }
        s = cudnnGetConvolutionBackwardFilterWorkspaceSize(
            p.m_ctx->m_cudnn, x.desc(), y.desc(), conv_desc_, filter_desc_,
            bwd_filter_algo_, &workspace);
        if(s)
            return cudnnGetErrorString(s);

        p.m_ctx->m_workspace.request(workspace);

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
      : CudnnOperation("convfwd"), ctx_(ctx), desc_(desc), x_(x), w_(w), y_(y)
    {
    }

    cudnnStatus_t exec(CudaProgram &p)
    {
        float alpha = 1.0f;
        float beta = 0.0f;

        return cudnnConvolutionForward(
            ctx_->m_cudnn, &alpha, x_->desc(), x_->deviceMem(),
            desc_->filter_desc_, w_->deviceMem(), desc_->conv_desc_,
            desc_->conv_fwd_algo_, ctx_->m_workspace.ptr(),
            ctx_->m_workspace.size(), &beta, y_->desc(), y_->deviceMem());
    }

    std::vector<std::shared_ptr<CudaTensor>> getInputs() const override
    {
        return {x_, w_};
    }

    std::vector<std::shared_ptr<CudaTensor>> getOutputs() const override
    {
        return {y_};
    }

    std::string info() const override
    {
        return convfwdalgostr(desc_->conv_fwd_algo_);
    }
};

struct CudnnConvolutionBwdBias : public CudnnOperation {
    const std::shared_ptr<CudaContext> ctx_;
    const std::shared_ptr<CudaTensor> dy_, db_;

    CudnnConvolutionBwdBias(std::shared_ptr<CudaContext> ctx,
                            std::shared_ptr<CudaTensor> dy,
                            std::shared_ptr<CudaTensor> db)
      : CudnnOperation("convbwdbias"), ctx_(ctx), dy_(dy), db_(db)
    {
    }

    cudnnStatus_t exec(CudaProgram &p)
    {
        float alpha = 1.0f;
        float beta = 0.0f;

        return cudnnConvolutionBackwardBias(ctx_->m_cudnn, &alpha, dy_->desc(),
                                            dy_->deviceMem(), &beta,
                                            db_->desc(), db_->deviceMem());
    }

    std::vector<std::shared_ptr<CudaTensor>> getInputs() const override
    {
        return {dy_};
    }

    std::vector<std::shared_ptr<CudaTensor>> getOutputs() const override
    {
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
      , ctx_(ctx)
      , desc_(desc)
      , x_(x)
      , dy_(dy)
      , dw_(dw)
    {
    }

    cudnnStatus_t exec(CudaProgram &p)
    {
        float alpha = 1.0f;
        float beta = 0.0f;

        return cudnnConvolutionBackwardFilter(
            ctx_->m_cudnn, &alpha, x_->desc(), x_->deviceMem(), dy_->desc(),
            dy_->deviceMem(), desc_->conv_desc_, desc_->bwd_filter_algo_,
            ctx_->m_workspace.ptr(), ctx_->m_workspace.size(), &beta,
            desc_->filter_desc_, dw_->deviceMem());
    }

    std::vector<std::shared_ptr<CudaTensor>> getInputs() const override
    {
        return {x_, dy_};
    }

    std::vector<std::shared_ptr<CudaTensor>> getOutputs() const override
    {
        return {dw_};
    }

    std::string info() const override
    {
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
                            std::shared_ptr<CudaTensor> dx, float dx_beta)
      : CudnnOperation("convbwddata")
      , ctx_(ctx)
      , desc_(desc)
      , w_(w)
      , dy_(dy)
      , dx_(dx)
      , dx_beta_(dx_beta)
    {
    }

    cudnnStatus_t exec(CudaProgram &p)
    {
        float alpha = 1.0f;

        return cudnnConvolutionBackwardData(
            ctx_->m_cudnn, &alpha, desc_->filter_desc_, w_->deviceMem(),
            dy_->desc(), dy_->deviceMem(), desc_->conv_desc_,
            desc_->bwd_data_algo_, ctx_->m_workspace.ptr(),
            ctx_->m_workspace.size(), &dx_beta_, dx_->desc(), dx_->deviceMem());
    }

    std::vector<std::shared_ptr<CudaTensor>> getInputs() const override
    {
        auto r = std::vector{w_, dy_};
        if(dx_beta_)
            r.push_back(dx_);
        return r;
    }

    std::vector<std::shared_ptr<CudaTensor>> getOutputs() const override
    {
        return {dx_};
    }

    std::string info() const override
    {
        std::stringstream ss;

        ss << convbwddataalgostr(desc_->bwd_data_algo_);
        if(dx_beta_) {
            ss << ", beta=" << dx_beta_;
        }
        return ss.str();
    }
};

static const char *
conv_setup(CudaProgram &p, const Node &n, bool training)
{
    auto x = p.lower_tensor(n.inputs_.get("x"));

    if(p.m_ctx->m_tensor_cores && x->data_type_ == Tensor::DataType::HALF &&
       !x->cpacked()) {
        assert(!x->m_grad);

        auto xx = std::make_shared<CudaTensor>(x->data_type_, x->dims_,
                                               CUDNN_TENSOR_NHWC, p.m_ctx,
                                               x->namePostfix("nhwc"));

        auto tr = std::make_shared<CudnnTransform>(x, xx, 0.0f);
        if(training)
            p.fwd(tr);
        else
            p.infer(tr);
        x = xx;
    }

    auto y = p.lower_tensor(n.outputs_.get("y"));
    auto w = p.lower_tensor(n.inputs_.get("w"));
    auto b = p.lower_tensor(n.inputs_.get("b"), 2);

    auto desc = std::make_shared<CudnnConvolutionDesc>();

    const int pad = n.attributes_.get("pad", 0);
    const int stride = n.attributes_.get("stride", 1);

    const char *err = desc->setup(p, *x, *w, *y, pad, stride, training);
    if(err)
        return err;

    if(!training) {
        p.infer(std::make_shared<CudnnConvolutionFwd>(p.m_ctx, desc, x, w, y));
        if(b)
            p.infer(std::make_shared<CudnnAddTensor>(b, y));
        return NULL;
    }

    p.fwd(std::make_shared<CudnnConvolutionFwd>(p.m_ctx, desc, x, w, y));
    if(b)
        p.fwd(std::make_shared<CudnnAddTensor>(b, y));

    auto dy = y->makeSharedGrad();

    if(b) {
        auto db = b->makePrivateGrad();
        p.bwd(std::make_shared<CudnnConvolutionBwdBias>(p.m_ctx, dy, db));
        p.upd(std::make_shared<CudnnAdam>(p, b, db));
    }

    auto dw = w->makePrivateGrad();
    p.bwd(
        std::make_shared<CudnnConvolutionBwdFilter>(p.m_ctx, desc, x, dy, dw));
    p.upd(std::make_shared<CudnnAdam>(p, w, dw));

    auto dx = x->makeSharedGrad();
    if(dx) {
        const float dx_beta = n.attributes_.get("dx.beta", 0.0f);
        p.bwd(std::make_shared<CudnnConvolutionBwdData>(p.m_ctx, desc, w, dy,
                                                        dx, dx_beta));
    }
    return NULL;
}

REGISTER_CUDA_OP("conv", conv_setup);

//------------------------------------------------------------------------

static const char *
activationalgo_to_str(const cudnnActivationDescriptor_t &desc)
{
    cudnnActivationMode_t mode;
    cudnnNanPropagation_t reluNanOpt;
    double coef;

    cudnnGetActivationDescriptor(desc, &mode, &reluNanOpt, &coef);

    switch(mode) {
    case CUDNN_ACTIVATION_SIGMOID:
        return "sigmoid";
    case CUDNN_ACTIVATION_RELU:
        return "relu";
    case CUDNN_ACTIVATION_TANH:
        return "tanh";
    case CUDNN_ACTIVATION_CLIPPED_RELU:
        return "clipped-relu";
    case CUDNN_ACTIVATION_ELU:
        return "elu";
    case CUDNN_ACTIVATION_IDENTITY:
        return "identity";
    case CUDNN_ACTIVATION_SWISH:
        return "swish";
    default:
        return "?";
    }
}

struct CudnnActivationFwd : public CudnnOperation {
    const std::shared_ptr<CudaTensor> x_, y_;
    const float y_beta_;

    cudnnActivationDescriptor_t desc_;

    CudnnActivationFwd(std::shared_ptr<CudaTensor> x,
                       std::shared_ptr<CudaTensor> y,
                       cudnnActivationMode_t mode, float alpha, float y_beta)
      : CudnnOperation("actfwd"), x_(x), y_(y), y_beta_(y_beta)
    {
        chkCUDNN(cudnnCreateActivationDescriptor(&desc_));
        chkCUDNN(cudnnSetActivationDescriptor(desc_, mode, CUDNN_PROPAGATE_NAN,
                                              alpha));
    }

    ~CudnnActivationFwd() { chkCUDNN(cudnnDestroyActivationDescriptor(desc_)); }

    cudnnStatus_t exec(CudaProgram &p)
    {
        float alpha = 1.0f;

        return cudnnActivationForward(p.m_ctx->m_cudnn, desc_, &alpha,
                                      x_->desc(), x_->deviceMem(), &y_beta_,
                                      y_->desc(), y_->deviceMem());
    }

    std::vector<std::shared_ptr<CudaTensor>> getInputs() const override
    {
        return {x_};
    }

    std::vector<std::shared_ptr<CudaTensor>> getOutputs() const override
    {
        return {y_};
    }

    std::string info() const override { return activationalgo_to_str(desc_); }
};

struct CudnnActivationBwd : public CudnnOperation {
    const std::shared_ptr<CudnnActivationFwd> fwd_;
    const std::shared_ptr<CudaTensor> x_, y_, dx_, dy_;
    const float dx_beta_;

    CudnnActivationBwd(const std::shared_ptr<CudnnActivationFwd> fwd,
                       std::shared_ptr<CudaTensor> x,
                       std::shared_ptr<CudaTensor> y,
                       std::shared_ptr<CudaTensor> dx,
                       std::shared_ptr<CudaTensor> dy, float dx_beta)
      : CudnnOperation("actbwd")
      , fwd_(fwd)
      , x_(x)
      , y_(y)
      , dx_(dx)
      , dy_(dy)
      , dx_beta_(dx_beta)
    {
    }

    cudnnStatus_t exec(CudaProgram &p)
    {
        float alpha = 1.0f;

        return cudnnActivationBackward(
            p.m_ctx->m_cudnn, fwd_->desc_, &alpha, y_->desc(), y_->deviceMem(),
            dy_->desc(), dy_->deviceMem(), x_->desc(), x_->deviceMem(),
            &dx_beta_, dx_->desc(), dx_->deviceMem());
    }

    std::vector<std::shared_ptr<CudaTensor>> getInputs() const override
    {
        auto r = std::vector{x_, y_, dy_};
        if(dx_beta_)
            r.push_back(dx_);
        return r;
    }

    std::vector<std::shared_ptr<CudaTensor>> getOutputs() const override
    {
        return {dx_};
    }
    std::string info() const override
    {
        return activationalgo_to_str(fwd_->desc_);
    }
};

static const char *
activation_setup(CudaProgram &p, const Node &n, bool training,
                 cudnnActivationMode_t mode, float alpha)
{
    auto x = p.lower_tensor(n.inputs_.get("x"));
    auto y = p.lower_tensor(n.outputs_.get("y"));
    auto fwd = std::make_shared<CudnnActivationFwd>(x, y, mode, alpha, 0);

    if(!training) {
        p.infer(fwd);
        return NULL;
    }

    p.fwd(fwd);

    auto dx = x->m_grad;
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

static const char *
sigmoid_setup(CudaProgram &p, const Node &n, bool training)
{
    return activation_setup(p, n, training, CUDNN_ACTIVATION_SIGMOID, 0.0f);
}

REGISTER_CUDA_OP("sigmoid", sigmoid_setup);

static const char *
tanh_setup(CudaProgram &p, const Node &n, bool training)
{
    return activation_setup(p, n, training, CUDNN_ACTIVATION_TANH, 0.0f);
}

REGISTER_CUDA_OP("tanh", tanh_setup);

//------------------------------------------------------------------------

struct CudaLeakyRelu : public CudaOperation {
    const std::shared_ptr<CudaTensor> x_, y_;
    const float alpha_;
    const size_t m_elements;

    CudaLeakyRelu(std::shared_ptr<CudaTensor> x, std::shared_ptr<CudaTensor> y,
                  float alpha)
      : CudaOperation("leakyrelu")
      , x_(x)
      , y_(y)
      , alpha_(alpha)
      , m_elements(x->dims_.elements())
    {
    }

    const char *exec(CudaProgram &p, long batch)
    {
        switch(x_->data_type_) {
        case Tensor::DataType::FLOAT:
            leaky_relu_float(m_elements, (float *)y_->deviceMem(),
                             (const float *)x_->deviceMem(), alpha_,
                             p.m_ctx->m_stream);
            break;
        default:
            return "LeakyRelu: Unsupported datatype";
        }
        return NULL;
    }

    std::vector<std::shared_ptr<CudaTensor>> getInputs() const override
    {
        return {x_};
    }

    std::vector<std::shared_ptr<CudaTensor>> getOutputs() const override
    {
        return {y_};
    }
};

static const char *
leakyrelu_setup(CudaProgram &p, const Node &n, bool training)
{
    if(training) {
        return "LeakRelu not supported for training";
    }

    auto x = p.lower_tensor(n.inputs_.get("x"));
    auto y = p.lower_tensor(n.outputs_.get("y"));
    float alpha = n.attributes_.get("alpha", 0.01f);
    auto op = std::make_shared<CudaLeakyRelu>(x, y, alpha);
    p.infer(op);
    return NULL;
}

REGISTER_CUDA_OP("leakyrelu", leakyrelu_setup);

//------------------------------------------------------------------------

struct CudnnPoolingFwd : public CudnnOperation {
    const std::shared_ptr<CudaTensor> x_, y_;

    cudnnPoolingDescriptor_t desc_;

    CudnnPoolingFwd(std::shared_ptr<CudaTensor> x,
                    std::shared_ptr<CudaTensor> y, cudnnPoolingMode_t mode,
                    int size, int pad, int stride)
      : CudnnOperation("poolfwd"), x_(x), y_(y)
    {
        chkCUDNN(cudnnCreatePoolingDescriptor(&desc_));

        chkCUDNN(cudnnSetPooling2dDescriptor(desc_, mode, CUDNN_PROPAGATE_NAN,
                                             size, size, pad, pad, stride,
                                             stride));
    }

    ~CudnnPoolingFwd() { chkCUDNN(cudnnDestroyPoolingDescriptor(desc_)); }

    cudnnStatus_t exec(CudaProgram &p)
    {
        float alpha = 1.0f;
        float beta = 0.0f;

        return cudnnPoolingForward(p.m_ctx->m_cudnn, desc_, &alpha, x_->desc(),
                                   x_->deviceMem(), &beta, y_->desc(),
                                   y_->deviceMem());
    }

    std::vector<std::shared_ptr<CudaTensor>> getInputs() const override
    {
        return {x_};
    }

    std::vector<std::shared_ptr<CudaTensor>> getOutputs() const override
    {
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
                    std::shared_ptr<CudaTensor> dy, float dx_beta)
      : CudnnOperation("poolbwd")
      , fwd_(fwd)
      , x_(x)
      , y_(y)
      , dx_(dx)
      , dy_(dy)
      , dx_beta_(dx_beta)
    {
    }

    cudnnStatus_t exec(CudaProgram &p)
    {
        float alpha = 1.0f;

        return cudnnPoolingBackward(
            p.m_ctx->m_cudnn, fwd_->desc_, &alpha, y_->desc(), y_->deviceMem(),
            dy_->desc(), dy_->deviceMem(), x_->desc(), x_->deviceMem(),
            &dx_beta_, dx_->desc(), dx_->deviceMem());
    }

    std::vector<std::shared_ptr<CudaTensor>> getInputs() const override
    {
        auto r = std::vector{x_, y_, dy_};
        if(dx_beta_)
            r.push_back(dx_);
        return r;
    }

    std::vector<std::shared_ptr<CudaTensor>> getOutputs() const override
    {
        return {dx_};
    }
};

static const char *
pooling_setup(CudaProgram &p, const Node &n, bool training,
              cudnnPoolingMode_t mode)
{
    auto x = p.lower_tensor(n.inputs_.get("x"));
    auto y = p.lower_tensor(n.outputs_.get("y"));

    int size;
    if(n.attributes_.get("global", false)) {
        size = x->dims_[2];
    } else {
        size = n.attributes_.get("size", 1);
    }
    const int pad = n.attributes_.get("pad", 0);
    const int stride = n.attributes_.get("stride", 1);

    auto fwd = std::make_shared<CudnnPoolingFwd>(x, y, mode, size, pad, stride);

    if(!training) {
        p.infer(fwd);
        return NULL;
    }

    p.fwd(fwd);

    auto dx = x->m_grad;
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
cublasErrStr(cublasStatus_t error)
{
    switch(error) {
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

    CudaGemm(cublasOperation_t transa, cublasOperation_t transb, int m, int n,
             int k, std::shared_ptr<CudaTensor> a, int lda,
             std::shared_ptr<CudaTensor> b, int ldb,
             std::shared_ptr<CudaTensor> c, int ldc)
      : CudaOperation("gemm")
      , transa_(transa)
      , transb_(transb)
      , m_(m)
      , n_(n)
      , k_(k)
      , a_(a)
      , lda_(lda)
      , b_(b)
      , ldb_(ldb)
      , c_(c)
      , ldc_(ldc)
    {
    }

    const char *exec(CudaProgram &p, long batch)
    {
        float alpha = 1.0f, beta = 0.0f;
        __half halpha = 1.0f, hbeta = 0.0f;

        cublasStatus_t s;
        switch(a_->m_type) {
        case CUDNN_DATA_FLOAT:

            s = cublasSgemm(p.m_ctx->m_cublas, transa_, transb_, m_, n_, k_,
                            &alpha, (const float *)a_->deviceMem(), lda_,
                            (const float *)b_->deviceMem(), ldb_, &beta,
                            (float *)c_->deviceMem(), ldc_);
            break;
        case CUDNN_DATA_HALF:
            s = cublasHgemm(p.m_ctx->m_cublas, transa_, transb_, m_, n_, k_,
                            &halpha, (const __half *)a_->deviceMem(), lda_,
                            (const __half *)b_->deviceMem(), ldb_, &hbeta,
                            (__half *)c_->deviceMem(), ldc_);
            break;
        default:
            return "Unsupported tensor datatype";
        }
        return cublasErrStr(s);
    }

    std::vector<std::shared_ptr<CudaTensor>> getInputs() const override
    {
        return {a_, b_};
    }

    std::vector<std::shared_ptr<CudaTensor>> getOutputs() const override
    {
        return {c_};
    }
};

static const char *
fc_setup(CudaProgram &p, const Node &n, bool training)
{
    auto x = p.lower_tensor(n.inputs_.get("x"));
    auto y = p.lower_tensor(n.outputs_.get("y"));

    auto w = p.lower_tensor(n.inputs_.get("w"));
    auto b = p.lower_tensor(n.inputs_.get("b"), 2);

    const bool transW = n.attributes_.get("transW", false);

    const cublasOperation_t transa = transW ? CUBLAS_OP_T : CUBLAS_OP_N;
    const cublasOperation_t transb = CUBLAS_OP_N;

    const int num_inputs = x->dims_[1];
    const int num_outputs = y->dims_[1];
    const int batch_size = x->dims_[0];

    auto fwd = std::make_shared<CudaGemm>(
        transa, transb, num_outputs, batch_size, num_inputs, w,
        transW ? num_inputs : num_outputs, x, num_inputs, y, num_outputs);

    if(!training) {
        p.infer(fwd);
        if(b)
            p.infer(std::make_shared<CudnnAddTensor>(b, y));
        return NULL;
    }

    p.fwd(fwd);

    if(b)
        p.fwd(std::make_shared<CudnnAddTensor>(b, y));

    if(!transW) {
        // Fix this
        return "fully connected with !transW not suppored";
    }

    auto dw = w->makePrivateGrad();
    auto dy = y->makeSharedGrad();

    p.bwd(std::make_shared<CudaGemm>(CUBLAS_OP_N, CUBLAS_OP_T, num_inputs,
                                     num_outputs, batch_size, x, num_inputs, dy,
                                     num_outputs, dw, num_inputs));
    p.upd(std::make_shared<CudnnAdam>(p, w, dw));

    if(b) {
        auto ones =
            p.lower_tensor(Tensor::make(x->data_type_, {batch_size, 1}, 1, 0));

        auto db = b->makePrivateGrad();
        p.bwd(std::make_shared<CudaGemm>(CUBLAS_OP_N, CUBLAS_OP_T, 1,
                                         num_outputs, batch_size, ones, 1, dy,
                                         num_outputs, db, 1));
        p.upd(std::make_shared<CudnnAdam>(p, b, db));
    }

    auto dx = x->m_grad;

    if(dx) {
        const float dx_beta = n.attributes_.get("dx.beta", 0.0f);
        if(dx_beta)
            return "dx_beta != 0";

        p.bwd(std::make_shared<CudaGemm>(CUBLAS_OP_N, CUBLAS_OP_N, num_inputs,
                                         batch_size, num_outputs, w, num_inputs,
                                         dy, num_outputs, dx, num_inputs));
    }
    return NULL;
}

REGISTER_CUDA_OP("fc", fc_setup);

//------------------------------------------------------------------------

struct CudaConvert : public CudaOperation {
    const std::shared_ptr<CudaTensor> x_, y_;
    const float scale_;
    const size_t m_elements;

    void (*algo_)(const void *src, void *dst, int elements, float scale,
                  cudaStream_t stream);

    CudaConvert(std::shared_ptr<CudaTensor> x, std::shared_ptr<CudaTensor> y,
                float scale)
      : CudaOperation("convert")
      , x_(x)
      , y_(y)
      , scale_(scale)
      , m_elements(x->dims_.elements())
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
        } else if(x_->data_type_ == Tensor::DataType::I16 &&
                  y_->data_type_ == Tensor::DataType::HALF) {
            algo_ = convert_i16_half;
        } else {
            algo_ = NULL;
        }
    }

    const char *exec(CudaProgram &p, long batch)
    {
        algo_(x_->deviceMem(), y_->deviceMem(), m_elements, scale_,
              p.m_ctx->m_stream);
        return NULL;
    }

    std::vector<std::shared_ptr<CudaTensor>> getInputs() const override
    {
        return {x_};
    }

    std::vector<std::shared_ptr<CudaTensor>> getOutputs() const override
    {
        return {y_};
    }
};

static const char *
convert_setup(CudaProgram &p, const Node &n, bool training)
{
    auto scale = n.attributes_.get("scale", 1.0f);

    auto xh = n.inputs_.get("x");
    auto x = p.lower_tensor(xh);

    auto yh = n.outputs_.get("y");

    if(xh->data_type_ == yh->data_type_ && scale == 1.0f) {
        auto y =
            std::make_shared<CudaTensor>(x, x->dims_, std::vector<int64_t>{},
                                         xh->namePostfix("nop-convert"));
        p.m_tensors[yh] = y;
        return NULL;
    }

    auto y = p.lower_tensor(yh, *x);
    auto op = std::make_shared<CudaConvert>(x, y, scale);

    if(op->algo_ == NULL) {
        return "Unable to convert between given formats";
    }

    if(!training) {
        p.infer(op);
    } else {
        p.fwd(op);
    }
    return NULL;
}

REGISTER_CUDA_OP("convert", convert_setup);

static std::shared_ptr<Node>
convert_fuse_nodes(CudaProgram &p, std::shared_ptr<Node> a,
                   std::shared_ptr<Node> b)
{
    float scale =
        a->attributes_.get("scale", 1.0f) * b->attributes_.get("scale", 1.0f);

    auto nn = std::make_shared<Node>("convert");
    nn->inputs_["x"] = a->inputs_.get("x");
    nn->outputs_["y"] = b->outputs_.get("y");
    nn->attributes_["scale"] = scale;
    return nn;
}

static Nodes
convert_transform(CudaProgram &p, const Nodes &input)
{
    Nodes nodes = input;

again:

    for(auto &n : nodes) {
        if(n->type_ != "convert")
            continue;

        auto it = nodes.findSingleDownStreamNode(n->y(), "convert");
        if(it != nodes.end()) {
            auto b = *it;
            n = convert_fuse_nodes(p, n, *it);
            nodes.erase(it);
            goto again;
        }
    }
    return nodes;
}

REGISTER_CUDA_TRANSFORM(200, CUDA_TRANSFORM_ALL, convert_transform);

//------------------------------------------------------------------------
struct CudaCatClassifierFwd : public CudaOperation {
    const std::shared_ptr<CudaTensor> x_, y_;

    CudaCatClassifierFwd(std::shared_ptr<CudaTensor> x,
                         std::shared_ptr<CudaTensor> y)
      : CudaOperation("catclassifierfwd"), x_(x), y_(y)
    {
    }

    const char *exec(CudaProgram &p, long batch)
    {
        switch(x_->m_type) {
        case CUDNN_DATA_FLOAT:
            catclassifier_fwd_float_i32(
                x_->dims_[0], (const float *)x_->deviceMem(),
                (int32_t *)y_->deviceMem(), x_->dims_[1], p.m_ctx->m_stream);
            break;
        case CUDNN_DATA_HALF:
            catclassifier_fwd_half_i32(
                x_->dims_[0], (const __half *)x_->deviceMem(),
                (int32_t *)y_->deviceMem(), x_->dims_[1], p.m_ctx->m_stream);
            break;
        default:
            return "Unsupported tensor datatype";
        }
        return NULL;
    }

    std::vector<std::shared_ptr<CudaTensor>> getInputs() const override
    {
        return {x_};
    }

    std::vector<std::shared_ptr<CudaTensor>> getOutputs() const override
    {
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
      , x_(x)
      , y_(y)
      , dx_(dx)
      , dy_(dy)
      , loss_(loss)
    {
    }

    const char *exec(CudaProgram &p, long batch)
    {
        const int n = x_->dims_[0];
        const int c = x_->dims_[1];
        const float scale = 1.0f / n;

        switch(x_->m_type) {
        case CUDNN_DATA_FLOAT:
            catclassifier_bwd_float_i32(
                n, (const float *)x_->deviceMem(), (float *)dx_->deviceMem(),
                (const int32_t *)y_->deviceMem(),
                (const int32_t *)dy_->deviceMem(),
                loss_ ? (float *)loss_->deviceMem() : NULL, c, scale,
                p.m_ctx->m_stream);

            break;
        case CUDNN_DATA_HALF:
            catclassifier_bwd_half_i32(
                n, (const __half *)x_->deviceMem(), (__half *)dx_->deviceMem(),
                (const int32_t *)y_->deviceMem(),
                (const int32_t *)dy_->deviceMem(),
                loss_ ? (float *)loss_->deviceMem() : NULL, c,
                scale * p.m_mp_scaling, p.m_ctx->m_stream);

            break;
        default:
            return "Unsupported tensor datatype";
        }
        return NULL;
    }

    std::vector<std::shared_ptr<CudaTensor>> getInputs() const override
    {
        return {x_, y_, dy_};
    }

    std::vector<std::shared_ptr<CudaTensor>> getOutputs() const override
    {
        return {dx_, loss_};
    }
};

static const char *
catclassifier_setup(CudaProgram &p, const Node &n, bool training)
{
    auto x = p.lower_tensor(n.inputs_.get("x"));
    auto y = p.lower_tensor(n.outputs_.get("y"));

    auto op = std::make_shared<CudaCatClassifierFwd>(x, y);

    if(!training) {
        p.infer(op);
        return NULL;
    }
    p.fwd(op);

    auto dx = x->m_grad;
    if(!dx)
        return NULL;
    auto dy = y->makeSharedGrad();
    auto loss = p.lower_tensor(n.outputs_.get("loss"));

    p.bwd(std::make_shared<CudaCatClassifierBwd>(x, y, dx, dy, loss));
    return NULL;
}

REGISTER_CUDA_OP("catclassifier", catclassifier_setup);

//------------------------------------------------------------------------
struct CudaMSEFwd : public CudaOperation {
    const std::shared_ptr<CudaTensor> x_, y_;
    const size_t m_elements;

    CudaMSEFwd(std::shared_ptr<CudaTensor> x, std::shared_ptr<CudaTensor> y)
      : CudaOperation("msefwd"), x_(x), y_(y), m_elements(x->dims_.elements())
    {
    }

    const char *exec(CudaProgram &p, long batch)
    {
        switch(x_->m_type) {
        case CUDNN_DATA_HALF:
            convert_half_float(x_->deviceMem(), y_->deviceMem(), m_elements,
                               1.0f, p.m_ctx->m_stream);
            break;
        default:
            return "Unsupported tensor datatype";
        }
        return NULL;
    }

    std::vector<std::shared_ptr<CudaTensor>> getInputs() const override
    {
        return {x_};
    }

    std::vector<std::shared_ptr<CudaTensor>> getOutputs() const override
    {
        return {y_};
    }
};

struct CudaMSEBwd : public CudaOperation {
    const std::shared_ptr<CudaTensor> x_, dx_, dy_, loss_;

    CudaMSEBwd(std::shared_ptr<CudaTensor> x, std::shared_ptr<CudaTensor> dx,
               std::shared_ptr<CudaTensor> dy, std::shared_ptr<CudaTensor> loss)
      : CudaOperation("msebwd"), x_(x), dx_(dx), dy_(dy), loss_(loss)
    {
    }

    const char *exec(CudaProgram &p, long batch)
    {
        const int n = x_->dims_[0];
        const int c = x_->dims_[1];
        const float scale = 1.0f / n;

        switch(x_->m_type) {
        case CUDNN_DATA_HALF:
            mse_bwd_half_float(n, (const __half *)x_->deviceMem(),
                               (__half *)dx_->deviceMem(),
                               (const float *)dy_->deviceMem(),
                               loss_ ? (float *)loss_->deviceMem() : NULL, c,
                               scale * p.m_mp_scaling, p.m_ctx->m_stream);

            break;
        default:
            return "Unsupported tensor datatype";
        }
        return NULL;
    }

    std::vector<std::shared_ptr<CudaTensor>> getInputs() const override
    {
        return {x_, dy_};
    }

    std::vector<std::shared_ptr<CudaTensor>> getOutputs() const override
    {
        return {dx_, loss_};
    }
};

static const char *
mse_setup(CudaProgram &p, const Node &n, bool training)
{
    auto x = p.lower_tensor(n.inputs_.get("x"));
    auto y = p.lower_tensor(n.outputs_.get("y"));

    auto op = std::make_shared<CudaMSEFwd>(x, y);

    if(!training) {
        p.infer(op);
        return NULL;
    }
    p.fwd(op);

    auto dx = x->m_grad;
    if(!dx)
        return NULL;
    auto dy = y->makeSharedGrad();
    auto loss = p.lower_tensor(n.outputs_.get("loss"));

    p.bwd(std::make_shared<CudaMSEBwd>(x, dx, dy, loss));
    return NULL;
}

REGISTER_CUDA_OP("mse", mse_setup);

//------------------------------------------------------------------------

struct CudnnDropoutFwd : public CudnnOperation {
    const std::shared_ptr<CudaContext> ctx_;
    const std::shared_ptr<CudaTensor> x_, y_;
    cudnnDropoutDescriptor_t desc_;
    size_t reserve_size_;
    void *reserve_;
    size_t states_size_;
    void *states_;

    CudnnDropoutFwd(CudaProgram &p, std::shared_ptr<CudaTensor> x,
                    std::shared_ptr<CudaTensor> y, float prob)
      : CudnnOperation("dropoutfwd"), ctx_(p.m_ctx), x_(x), y_(y)
    {
        chkCUDNN(cudnnDropoutGetReserveSpaceSize(x_->desc(), &reserve_size_));
        chkCuda(cudaMalloc(&reserve_, reserve_size_));

        chkCUDNN(cudnnDropoutGetStatesSize(p.m_ctx->m_cudnn, &states_size_));
        chkCuda(cudaMalloc(&states_, states_size_));

        chkCUDNN(cudnnCreateDropoutDescriptor(&desc_));
        chkCUDNN(cudnnSetDropoutDescriptor(desc_, p.m_ctx->m_cudnn, prob,
                                           states_, states_size_, 0));
    }

    ~CudnnDropoutFwd()
    {
        chkCUDNN(cudnnDestroyDropoutDescriptor(desc_));

        chkCuda(cudaFree(states_));
        chkCuda(cudaFree(reserve_));
    }

    cudnnStatus_t exec(CudaProgram &p)
    {
        return cudnnDropoutForward(p.m_ctx->m_cudnn, desc_, x_->desc(),
                                   x_->deviceMem(), y_->desc(), y_->deviceMem(),
                                   reserve_, reserve_size_);
    }

    std::vector<std::shared_ptr<CudaTensor>> getInputs() const override
    {
        return {x_};
    }

    std::vector<std::shared_ptr<CudaTensor>> getOutputs() const override
    {
        return {y_};
    }
};

struct CudnnDropoutBwd : public CudnnOperation {
    const std::shared_ptr<CudnnDropoutFwd> fwd_;
    const std::shared_ptr<CudaTensor> dx_, dy_;

    CudnnDropoutBwd(std::shared_ptr<CudnnDropoutFwd> fwd,
                    std::shared_ptr<CudaTensor> dx,
                    std::shared_ptr<CudaTensor> dy)
      : CudnnOperation("dropoutbwd"), fwd_(fwd), dx_(dx), dy_(dy)
    {
    }

    ~CudnnDropoutBwd() {}

    cudnnStatus_t exec(CudaProgram &p)
    {
        return cudnnDropoutBackward(
            p.m_ctx->m_cudnn, fwd_->desc_, dy_->desc(), dy_->deviceMem(),
            dx_->desc(), dx_->deviceMem(), fwd_->reserve_, fwd_->reserve_size_);
    }

    std::vector<std::shared_ptr<CudaTensor>> getInputs() const override
    {
        return {dy_};
    }

    std::vector<std::shared_ptr<CudaTensor>> getOutputs() const override
    {
        return {dx_};
    }
};

static const char *
dropout_setup(CudaProgram &p, const Node &n, bool training)
{
    assert(training);

    auto x = p.lower_tensor(n.inputs_.get("x"));
    auto y = p.lower_tensor(n.outputs_.get("y"));
    const float prob = n.attributes_.get("prob", 0.5f);

    auto fwd = std::make_shared<CudnnDropoutFwd>(p, x, y, prob);
    p.fwd(fwd);

    auto dx = x->m_grad;
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
    auto ly = p.m_tensors[y];

    if(ly) {
        auto x = n->inputs_.get("x");
        auto lx = std::make_shared<CudaTensor>(ly->m_storage, ly->dims_,
                                               p.tensorFormat(*ly),
                                               ly->namePostfix("dropout"));
        p.m_tensors[x] = ly;

    } else {
        auto x = p.lower_tensor(n->inputs_.get("x"));
        ly = std::make_shared<CudaTensor>(x->m_storage, x->dims_,
                                          p.tensorFormat(*x),
                                          x->namePostfix("dropout"));
        p.m_tensors[y] = ly;
    }
    return {};
}

REGISTER_CUDA_OP("dropout", dropout_setup);

static Nodes
dropout_transform(CudaProgram &p, const Nodes &nodes)
{
    Nodes r;

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
                            std::shared_ptr<CudaTensor> y, float epsilon)
      : CudnnOperation("bninf")
      , x_(x)
      , s_(s)
      , b_(b)
      , m_(m)
      , v_(v)
      , y_(y)
      , epsilon_(epsilon)
    {
    }

    cudnnStatus_t exec(CudaProgram &p)
    {
        float alpha = 1.0f;
        float beta = 0.0f;

        return cudnnBatchNormalizationForwardInference(
            p.m_ctx->m_cudnn, CUDNN_BATCHNORM_SPATIAL, &alpha, &beta,
            x_->desc(), x_->deviceMem(), y_->desc(), y_->deviceMem(),
            s_->desc(), s_->deviceMem(), b_->deviceMem(), m_->deviceMem(),
            v_->deviceMem(), epsilon_);
    }

    std::vector<std::shared_ptr<CudaTensor>> getInputs() const override
    {
        return {x_, s_, b_, m_, v_};
    }

    std::vector<std::shared_ptr<CudaTensor>> getOutputs() const override
    {
        return {y_};
    }
};

struct CudnnBatchNormTrain : public CudnnOperation {
    const std::shared_ptr<CudaTensor> x_, s_, b_, m_, v_, y_, sm_, sv_;
    const float epsilon_;
    const float expavgf_;

    CudnnBatchNormTrain(
        std::shared_ptr<CudaTensor> x, std::shared_ptr<CudaTensor> s,
        std::shared_ptr<CudaTensor> b, std::shared_ptr<CudaTensor> m,
        std::shared_ptr<CudaTensor> v, std::shared_ptr<CudaTensor> y,
        std::shared_ptr<CudaTensor> sm, std::shared_ptr<CudaTensor> sv,
        float epsilon, float expavgf)
      : CudnnOperation("bntrain")
      , x_(x)
      , s_(s)
      , b_(b)
      , m_(m)
      , v_(v)
      , y_(y)
      , sm_(sm)
      , sv_(sv)
      , epsilon_(epsilon)
      , expavgf_(expavgf)
    {
    }

    cudnnStatus_t exec(CudaProgram &p)
    {
        float alpha = 1.0f;
        float beta = 0.0f;

        return cudnnBatchNormalizationForwardTraining(
            p.m_ctx->m_cudnn, CUDNN_BATCHNORM_SPATIAL, &alpha, &beta,
            x_->desc(), x_->deviceMem(), y_->desc(), y_->deviceMem(),
            s_->desc(), s_->deviceMem(), b_->deviceMem(), expavgf_,
            m_->deviceMem(), v_->deviceMem(), epsilon_, sm_->deviceMem(),
            sv_->deviceMem());
    }

    std::vector<std::shared_ptr<CudaTensor>> getInputs() const override
    {
        return {x_, s_, b_, m_, v_};
    }

    std::vector<std::shared_ptr<CudaTensor>> getOutputs() const override
    {
        return {y_, sm_, sv_};
    }
};

struct CudnnBatchNormBwd : public CudnnOperation {
    const std::shared_ptr<CudaTensor> x_, dy_, dx_, s_, ds_, db_, sm_, sv_;
    const float epsilon_;
    const float dx_beta_;

    CudnnBatchNormBwd(CudnnBatchNormTrain &fwd, std::shared_ptr<CudaTensor> dy,
                      std::shared_ptr<CudaTensor> dx,
                      std::shared_ptr<CudaTensor> ds,
                      std::shared_ptr<CudaTensor> db, float dx_beta)
      : CudnnOperation("bnbwd")
      , x_(fwd.x_)
      , dy_(dy)
      , dx_(dx)
      , s_(fwd.s_)
      , ds_(ds)
      , db_(db)
      , sm_(fwd.sm_)
      , sv_(fwd.sv_)
      , epsilon_(fwd.epsilon_)
      , dx_beta_(dx_beta)
    {
    }

    cudnnStatus_t exec(CudaProgram &p)
    {
        float alpha = 1.0f, beta = 0.0f;

        return cudnnBatchNormalizationBackward(
            p.m_ctx->m_cudnn, CUDNN_BATCHNORM_SPATIAL, &alpha, &dx_beta_,
            &alpha, &beta, x_->desc(), x_->deviceMem(), dy_->desc(),
            dy_->deviceMem(), dx_->desc(), dx_->deviceMem(), s_->desc(),
            s_->deviceMem(), ds_->deviceMem(), db_->deviceMem(), epsilon_,
            sm_->deviceMem(), sv_->deviceMem());
    }

    std::vector<std::shared_ptr<CudaTensor>> getInputs() const override
    {
        auto r = std::vector{x_, dy_, s_, sm_, sv_};
        if(dx_beta_)
            r.push_back(dx_);
        return r;
    }

    std::vector<std::shared_ptr<CudaTensor>> getOutputs() const override
    {
        return {dx_, ds_, db_};
    }
};

static const char *
batchnorm_setup(CudaProgram &p, const Node &n, bool training)
{
    auto x = p.lower_tensor(n.inputs_.get("x"));
    auto y = p.lower_tensor(n.outputs_.get("y"));

    auto s = p.lower_tensor(n.inputs_.get("s"), 2);
    auto b = p.lower_tensor(n.inputs_.get("b"), 2);
    auto m = p.lower_tensor(n.inputs_.get("m"), 2);
    auto v = p.lower_tensor(n.inputs_.get("v"), 2);
    const float epsilon = n.attributes_.get("epsilon", 1e-5f);

    if(!training) {
        p.infer(std::make_shared<CudnnBatchNormInference>(x, s, b, m, v, y,
                                                          epsilon));
        return NULL;
    }

    auto sm = std::make_shared<CudaTensor>(*m, m->namePostfix("smean"));
    auto sv = std::make_shared<CudaTensor>(*v, v->namePostfix("svar"));

    // expavgf, 0 = mean and var is stationary, 1 = overwritten
    const float expavgf = n.attributes_.get("expavgf", 0.1f);
    const float dx_beta = n.attributes_.get("dx.beta", 0.0f);
    auto f = std::make_shared<CudnnBatchNormTrain>(x, s, b, m, v, y, sm, sv,
                                                   epsilon, expavgf);
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
        case CUDNN_OP_TENSOR_ADD:
            return "add";
        case CUDNN_OP_TENSOR_MUL:
            return "mul";
        case CUDNN_OP_TENSOR_MIN:
            return "min";
        case CUDNN_OP_TENSOR_MAX:
            return "max";
        case CUDNN_OP_TENSOR_SQRT:
            return "sqrt";
        case CUDNN_OP_TENSOR_NOT:
            return "not";
        default:
            abort();
        }
    }

    CudnnOpTensor(std::shared_ptr<CudaTensor> a, std::shared_ptr<CudaTensor> b,
                  std::shared_ptr<CudaTensor> c, cudnnOpTensorOp_t op)
      : CudnnOperation(opname(op)), a_(a), b_(b), c_(c)
    {
        cudnnCreateOpTensorDescriptor(&desc_);
        cudnnSetOpTensorDescriptor(desc_, op, CUDNN_DATA_FLOAT,
                                   CUDNN_PROPAGATE_NAN);
    }

    ~CudnnOpTensor() { cudnnDestroyOpTensorDescriptor(desc_); }

    cudnnStatus_t exec(CudaProgram &p)
    {
        float alpha = 1.0f;
        float beta = 0.0f;

        return cudnnOpTensor(p.m_ctx->m_cudnn, desc_, &alpha, a_->desc(),
                             a_->deviceMem(), &alpha, b_->desc(),
                             b_->deviceMem(), &beta, c_->desc(),
                             c_->deviceMem());
    }

    std::vector<std::shared_ptr<CudaTensor>> getInputs() const override
    {
        return {a_, b_};
    }

    std::vector<std::shared_ptr<CudaTensor>> getOutputs() const override
    {
        return {c_};
    }
};

static const char *
sum_setup(CudaProgram &p, const Node &n, bool training)
{
    auto x0 = p.lower_tensor(n.inputs_.get("x0"));
    auto x1 = p.lower_tensor(n.inputs_.get("x1"));

    auto y = p.lower_tensor(n.outputs_.get("y"));

    auto fwd = std::make_shared<CudnnOpTensor>(x0, x1, y, CUDNN_OP_TENSOR_ADD);

    if(!training) {
        p.infer(fwd);
        return NULL;
    }

    p.fwd(fwd);

    auto dy = y->makeSharedGrad();

    auto dx0 = x0->m_grad;
    if(dx0) {
        const float beta = n.attributes_.get("dx0.beta", 0.0f);
        p.bwd(std::make_shared<CudnnTransform>(dy, dx0, beta));
    }
    auto dx1 = x1->m_grad;
    if(dx1) {
        const float beta = n.attributes_.get("dx1.beta", 0.0f);
        p.bwd(std::make_shared<CudnnTransform>(dy, dx1, beta));
    }
    return NULL;
}

REGISTER_CUDA_OP("sum", sum_setup);

static const char *
add_setup(CudaProgram &p, const Node &n, bool training)
{
    auto x = p.lower_tensor(n.inputs_.get("x"));
    auto y = p.lower_tensor(n.outputs_.get("y"));
    auto b = p.lower_tensor(n.inputs_.get("b"));

    auto fwd = std::make_shared<CudnnOpTensor>(x, b, y, CUDNN_OP_TENSOR_ADD);

    if(!training) {
        p.infer(fwd);
        return NULL;
    }
    return "Add not supported for backprop (yet)";
}

REGISTER_CUDA_OP("add", add_setup);

//------------------------------------------------------------------------

struct CudnnSoftmaxFwd : public CudnnOperation {
    const std::shared_ptr<CudaTensor> x_, y_;

    CudnnSoftmaxFwd(std::shared_ptr<CudaTensor> x,
                    std::shared_ptr<CudaTensor> y)
      : CudnnOperation("softmaxfwd"), x_(x), y_(y)
    {
    }

    cudnnStatus_t exec(CudaProgram &p)
    {
        float alpha = 1.0f, beta = 0.0f;

        return cudnnSoftmaxForward(p.m_ctx->m_cudnn, CUDNN_SOFTMAX_ACCURATE,
                                   CUDNN_SOFTMAX_MODE_CHANNEL, &alpha,
                                   x_->desc(), x_->deviceMem(), &beta,
                                   y_->desc(), y_->deviceMem());
    }

    std::vector<std::shared_ptr<CudaTensor>> getInputs() const override
    {
        return {x_};
    }

    std::vector<std::shared_ptr<CudaTensor>> getOutputs() const override
    {
        return {y_};
    }
};

static const char *
softmax_setup(CudaProgram &p, const Node &n, bool training)
{
    if(training)
        return "not supported for training";

    auto x = p.lower_tensor(n.inputs_.get("x"));
    auto y = p.lower_tensor(n.outputs_.get("y"));
    p.infer(std::make_shared<CudnnSoftmaxFwd>(x, y));
    return NULL;
}

REGISTER_CUDA_OP("softmax", softmax_setup);

//------------------------------------------------------------------------

static const char *
bnopsstr(cudnnBatchNormOps_t ops)
{
    switch(ops) {
    case CUDNN_BATCHNORM_OPS_BN:
        return "bn";
    case CUDNN_BATCHNORM_OPS_BN_ACTIVATION:
        return "bn_act";
    case CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION:
        return "bn_add_act";
    default:
        return "??";
    }
}

struct CudnnBatchNormActTrain : public CudnnOperation {
    const std::shared_ptr<CudaContext> ctx_;
    const std::shared_ptr<CudaTensor> x_, z_, s_, b_, m_, v_, y_, sm_, sv_;
    const float epsilon_;
    const float expavgf_;
    const cudnnBatchNormOps_t ops_;
    const cudnnBatchNormMode_t mode_;
    cudnnActivationDescriptor_t desc_;

    void *reserve_;
    size_t reserve_size_;

    CudnnBatchNormActTrain(
        CudaProgram &p, std::shared_ptr<CudaTensor> x,
        std::shared_ptr<CudaTensor> z, std::shared_ptr<CudaTensor> s,
        std::shared_ptr<CudaTensor> b, std::shared_ptr<CudaTensor> m,
        std::shared_ptr<CudaTensor> v, std::shared_ptr<CudaTensor> y,
        std::shared_ptr<CudaTensor> sm, std::shared_ptr<CudaTensor> sv,
        float epsilon, float expavgf, cudnnBatchNormOps_t ops,
        cudnnActivationMode_t activation_mode, float actalpha)
      : CudnnOperation(std::string(bnopsstr(ops)) + "_fwd.persistent")
      , ctx_(p.m_ctx)
      , x_(x)
      , z_(z)
      , s_(s)
      , b_(b)
      , m_(m)
      , v_(v)
      , y_(y)
      , sm_(sm)
      , sv_(sv)
      , epsilon_(epsilon)
      , expavgf_(expavgf)
      , ops_(ops)
      , mode_(CUDNN_BATCHNORM_SPATIAL_PERSISTENT)
    {
        chkCUDNN(cudnnCreateActivationDescriptor(&desc_));
        chkCUDNN(cudnnSetActivationDescriptor(desc_, activation_mode,
                                              CUDNN_PROPAGATE_NAN, actalpha));

        chkCUDNN(cudnnGetBatchNormalizationTrainingExReserveSpaceSize(
            p.m_ctx->m_cudnn, mode_, ops_, desc_, x_->desc(), &reserve_size_));
        chkCuda(cudaMalloc(&reserve_, reserve_size_));
    }

    ~CudnnBatchNormActTrain()
    {
        chkCUDNN(cudnnDestroyActivationDescriptor(desc_));
        chkCuda(cudaFree(reserve_));
    }

    cudnnStatus_t exec(CudaProgram &p)
    {
        float alpha = 1.0f;
        float beta = 0.0f;

        return cudnnBatchNormalizationForwardTrainingEx(
            ctx_->m_cudnn, mode_, ops_, &alpha, &beta, x_->desc(),
            x_->deviceMem(), z_ ? z_->desc() : NULL,
            z_ ? z_->deviceMem() : NULL, y_->desc(), y_->deviceMem(),
            s_->desc(), s_->deviceMem(), b_->deviceMem(), expavgf_,
            m_->deviceMem(), v_->deviceMem(), epsilon_, sm_->deviceMem(),
            sv_->deviceMem(), desc_, ctx_->m_workspace.ptr(),
            ctx_->m_workspace.size(), reserve_, reserve_size_);
    }

    std::vector<std::shared_ptr<CudaTensor>> getInputs() const override
    {
        auto r = std::vector{x_, s_, b_, m_, v_};
        if(z_)
            r.push_back(z_);
        return r;
    }

    std::vector<std::shared_ptr<CudaTensor>> getOutputs() const override
    {
        return {y_, sm_, sv_, m_, v_};
    }
};

struct CudnnBatchNormActBwd : public CudnnOperation {
    const std::shared_ptr<CudnnBatchNormActTrain> fwd_;
    std::shared_ptr<CudaTensor> dy_, dx_, dz_, ds_, db_;
    cudnnBatchNormOps_t ops_;
    const float dx_beta_;

    CudnnBatchNormActBwd(
        CudaProgram &p, std::shared_ptr<CudnnBatchNormActTrain> fwd,
        std::shared_ptr<CudaTensor> dy, std::shared_ptr<CudaTensor> dx,
        std::shared_ptr<CudaTensor> dz, std::shared_ptr<CudaTensor> ds,
        std::shared_ptr<CudaTensor> db, cudnnBatchNormOps_t ops, float dx_beta)
      : CudnnOperation(std::string(bnopsstr(ops)) + "_bwd.persistent")
      , fwd_(fwd)
      , dy_(dy)
      , dx_(dx)
      , dz_(dz)
      , ds_(ds)
      , db_(db)
      , ops_(ops)
      , dx_beta_(dx_beta)
    {
        size_t workspace;
        chkCUDNN(cudnnGetBatchNormalizationBackwardExWorkspaceSize(
            p.m_ctx->m_cudnn, fwd->mode_, fwd->ops_, fwd->x_->desc(),
            fwd->y_->desc(), dy_->desc(), dz_ ? dz_->desc() : NULL, dx_->desc(),
            ds_->desc(), fwd->desc_, &workspace));
        p.m_ctx->m_workspace.request(workspace);
    }

    cudnnStatus_t exec(CudaProgram &p)
    {
        float alpha = 1.0f, beta = 0.0f;

        return cudnnBatchNormalizationBackwardEx(
            p.m_ctx->m_cudnn, fwd_->mode_, ops_, &alpha, &dx_beta_, &alpha,
            &beta, fwd_->x_->desc(), fwd_->x_->deviceMem(), fwd_->y_->desc(),
            fwd_->y_->deviceMem(), dy_->desc(), dy_->deviceMem(),
            dz_ ? dz_->desc() : NULL, dz_ ? dz_->deviceMem() : NULL,
            dx_->desc(), dx_->deviceMem(), fwd_->s_->desc(),
            fwd_->s_->deviceMem(), fwd_->b_->deviceMem(), ds_->deviceMem(),
            db_->deviceMem(), fwd_->epsilon_, fwd_->sm_->deviceMem(),
            fwd_->sv_->deviceMem(), fwd_->desc_, p.m_ctx->m_workspace.ptr(),
            p.m_ctx->m_workspace.size(), fwd_->reserve_, fwd_->reserve_size_);
    }

    std::vector<std::shared_ptr<CudaTensor>> getInputs() const override
    {
        auto r = std::vector{fwd_->x_, fwd_->y_,  dy_,      fwd_->s_,
                             fwd_->b_, fwd_->sm_, fwd_->sv_};
        if(dx_beta_)
            r.push_back(dx_);
        return r;
    }

    std::vector<std::shared_ptr<CudaTensor>> getOutputs() const override
    {
        auto r = std::vector{dx_, ds_, db_};
        if(dz_)
            r.push_back(dz_);
        return r;
    }

    virtual bool killOutput(std::shared_ptr<CudaTensorStorage> s)
    {
        if(dz_ && dz_->m_storage == s) {
            dz_.reset();
            assert(ops_ == CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION);
            ops_ = CUDNN_BATCHNORM_OPS_BN_ACTIVATION;
            m_kind = std::string(bnopsstr(ops_)) + "_bwd.persistent";
            return true;
        }
        return false;
    }
};

static const char *
batchnorm_persistent_setup(CudaProgram &p, const Node &n, bool training)
{
    if(!training)
        return "not supported for inferenece";

    auto x0 = p.lower_tensor(n.inputs_.get("x0"));
    auto x1 = p.lower_tensor(n.inputs_.get("x1"));
    auto y = p.lower_tensor(n.outputs_.get("y"));

    auto s = p.lower_tensor(n.inputs_.get("s"), 2);
    auto b = p.lower_tensor(n.inputs_.get("b"), 2);
    auto m = p.lower_tensor(n.inputs_.get("m"), 2);
    auto v = p.lower_tensor(n.inputs_.get("v"), 2);
    const float epsilon = n.attributes_.get("epsilon", 1e-5f);

    auto sm = std::make_shared<CudaTensor>(*m, m->namePostfix("smean"));
    auto sv = std::make_shared<CudaTensor>(*v, v->namePostfix("svar"));

    const float expavgf = n.attributes_.get("expavgf", 0.1f);

    auto ops = CUDNN_BATCHNORM_OPS_BN;
    if(n.attributes_.get("relu", false)) {
        ops = CUDNN_BATCHNORM_OPS_BN_ACTIVATION;
        if(x1)
            ops = CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION;
    } else {
        assert(x1 == nullptr);
    }

    auto activation_mode = CUDNN_ACTIVATION_RELU;
    float activation_alpha = 0.0f;

    auto f = std::make_shared<CudnnBatchNormActTrain>(
        p, x0, x1, s, b, m, v, y, sm, sv, epsilon, expavgf, ops,
        activation_mode, activation_alpha);
    p.fwd(f);

    auto dx0 = x0->makeSharedGrad();
    auto dx1 = x1 ? x1->makeSharedGrad() : nullptr;
    auto dy = y->makeSharedGrad();
    auto ds = s->makePrivateGrad();
    auto db = b->makePrivateGrad();

    const float dx0_beta = n.attributes_.get("dx0.beta", 0.0f);
    const float dx1_beta = n.attributes_.get("dx1.beta", 0.0f);

    if(dx0_beta) {
        return "dx0.beta is non-zero";
    }
    if(dx1_beta) {
        return "dx1.beta is non-zero";
    }

    p.bwd(std::make_shared<CudnnBatchNormActBwd>(p, f, dy, dx0, dx1, ds, db,
                                                 ops, 0.0f));

    p.upd(std::make_shared<CudnnAdam>(p, s, ds));
    p.upd(std::make_shared<CudnnAdam>(p, b, db));
    return NULL;
}

REGISTER_CUDA_OP("batchnorm.persistent", batchnorm_persistent_setup);

static std::shared_ptr<Node>
batchnorm_persistent_transform_node(CudaProgram &p, std::shared_ptr<Node> bn,
                                    std::shared_ptr<Node> sum,
                                    std::shared_ptr<Node> relu)
{
    auto x = bn->inputs_["x"];

    if(x->data_type_ != Tensor::DataType::HALF)
        return nullptr;

    if(x->dims_[1] % 4)
        return nullptr;

    auto lx = p.m_tensors.find(x);
    if(lx != p.m_tensors.end()) {
        if(!lx->second->cpacked())
            return nullptr;
    }

    auto y = relu ? relu->outputs_["y"] : bn->outputs_["y"];

    auto ly = p.m_tensors.find(y);
    if(ly != p.m_tensors.end()) {
        if(!ly->second->cpacked())
            return nullptr;
    }

    auto nn = std::make_shared<Node>("batchnorm.persistent");

    nn->inputs_["x0"] = bn->inputs_["x"];
    auto bn_y = bn->outputs_["y"];

    if(sum) {
        auto sum_x0 = sum->inputs_["x0"];
        auto sum_x1 = sum->inputs_["x1"];

        if(sum_x0 == bn_y) {
            nn->inputs_["x1"] = sum->inputs_["x1"];
        } else {
            nn->inputs_["x1"] = sum->inputs_["x0"];
        }
    }
    nn->inputs_["s"] = bn->inputs_["s"];
    nn->inputs_["b"] = bn->inputs_["b"];
    nn->inputs_["m"] = bn->inputs_["m"];
    nn->inputs_["v"] = bn->inputs_["v"];

    if(bn->attributes_.find("epsilon") != bn->attributes_.end())
        nn->attributes_["epsilon"] = bn->attributes_["epsilon"];

    if(relu)
        nn->attributes_["relu"] = true;

    nn->outputs_["y"] = y;
    return nn;
}

static Nodes
batchnorm_relu_transform(CudaProgram &p, const Nodes &nodes)
{
    Nodes r;
    const ssize_t num_nodes = nodes.size();

    for(ssize_t i = 0; i < num_nodes; i++) {
        std::shared_ptr<Node> n = nodes[i];

        if(i < num_nodes - 1 && nodes[i + 0]->type_ == "batchnorm" &&
           nodes[i + 1]->type_ == "relu") {
            auto n2 = batchnorm_persistent_transform_node(p, nodes[i], nullptr,
                                                          nodes[i + 1]);
            if(n2) {
                i++;
                n = n2;
            }
        } else if(i < num_nodes - 2 && nodes[i + 0]->type_ == "batchnorm" &&
                  nodes[i + 1]->type_ == "sum" &&
                  nodes[i + 2]->type_ == "relu") {
            auto n2 = batchnorm_persistent_transform_node(
                p, nodes[i], nodes[i + 1], nodes[i + 2]);
            if(n2) {
                i += 2;
                n = n2;
            }
        } else if(nodes[i]->type_ == "batchnorm") {
            auto n2 = batchnorm_persistent_transform_node(p, nodes[i], nullptr,
                                                          nullptr);
            if(n2) {
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
      , x_(x)
      , theta_(theta)
      , y_(y)
      , grid_(std::make_shared<CudaTensor>(
            Tensor::DataType::FLOAT,
            Dims{y_->dims_[0], 2, y_->dims_[2], y_->dims_[3]},
            CUDNN_TENSOR_NHWC, ctx))
    {
        int dims[4] = {
            (int)y_->dims_[0],  // n
            1,                  // c
            (int)y_->dims_[2],  // h
            (int)y_->dims_[3]   // w
        };
        chkCUDNN(cudnnCreateSpatialTransformerDescriptor(&desc_));
        chkCUDNN(cudnnSetSpatialTransformerNdDescriptor(
            desc_, CUDNN_SAMPLER_BILINEAR, CUDNN_DATA_FLOAT, 4, dims));
        grid_->m_storage->alloc();
    }

    ~CudnnSpatialTransformFwd()
    {
        chkCUDNN(cudnnDestroySpatialTransformerDescriptor(desc_));
    }

    cudnnStatus_t exec(CudaProgram &p)
    {
        float alpha = 1.0f, beta = 0.0f;
        cudnnStatus_t s;

        s = cudnnSpatialTfGridGeneratorForward(
            p.m_ctx->m_cudnn, desc_, theta_->deviceMem(), grid_->deviceMem());
        if(s)
            return s;
        return cudnnSpatialTfSamplerForward(
            p.m_ctx->m_cudnn, desc_, &alpha, x_->desc(), x_->deviceMem(),
            grid_->deviceMem(), &beta, y_->desc(), y_->deviceMem());
    }

    std::vector<std::shared_ptr<CudaTensor>> getInputs() const override
    {
        return {x_, theta_};
    }

    std::vector<std::shared_ptr<CudaTensor>> getOutputs() const override
    {
        return {y_};
    }
};

static const char *
spatialtransform_setup(CudaProgram &p, const Node &n, bool training)
{
    auto x = p.lower_tensor(n.inputs_.get("x"));
    auto y = p.lower_tensor(n.outputs_.get("y"));

    const bool disable = !training && !n.attributes_.get("inference", false);
    const bool bypass = disable && x->dims_ == y->dims_;

    if(bypass) {
        p.infer(std::make_shared<CudnnTransform>(x, y, 0.0f));
        return NULL;
    }

    auto th = n.inputs_.get("theta");

    std::shared_ptr<CudaTensor> theta;

    if(disable || !th) {
        th = makeCPUTensor(Tensor::DataType::FLOAT,
                           Dims({p.m_batch_size, 2, 3}), "theta.identity");
        auto ta = th->access();
        for(int i = 0; i < p.m_batch_size; i++) {
            ta->set({i, 0, 0}, 1);
            ta->set({i, 1, 1}, 1);
        }
    }
    theta = p.lower_tensor(th);

    auto op = std::make_shared<CudnnSpatialTransformFwd>(p.m_ctx, x, theta, y);

    if(training) {
        p.fwd(op);
    } else {
        p.infer(op);
    }
    return NULL;
}

REGISTER_CUDA_OP("spatialtransform", spatialtransform_setup);

static Nodes
spatialtransform_transform(CudaProgram &p, const Nodes &nodes)
{
    Nodes r;

    for(size_t i = 0; i < nodes.size(); i++) {
        auto &n = nodes[i];
        if(n->type_ == "spatialtransform") {
            auto x = n->inputs_.get("x");
            if(x && x->data_type_ != Tensor::DataType::FLOAT) {
                Tensors emptyset;
                auto n0 = Node::make(
                    "convert", {{"x", x}},
                    {{"datatype", (int)Tensor::DataType::FLOAT}}, emptyset)[0];
                auto n1 = Node::make(
                    "spatialtransform",
                    {{"x", n0->y()}, {"theta", n->inputs_.get("theta")}},
                    n->attributes_, emptyset)[0];
                auto n2 =
                    Node::make("convert", {{"x", n1->y()}},
                               {{"datatype", (int)x->data_type_}}, emptyset)[0];
                n2->outputs_["y"] = n->outputs_.get("y");
                r.push_back(n0);
                r.push_back(n1);
                r.push_back(n2);
                continue;
            }
        }
        r.push_back(n);
    }
    return r;
}

REGISTER_CUDA_TRANSFORM(120, CUDA_TRANSFORM_ALL, spatialtransform_transform);

//------------------------------------------------------------------------

static std::vector<std::shared_ptr<Node>>
reshape_transform_node(CudaProgram &p, std::shared_ptr<Node> n)
{
    auto x = p.lower_tensor(n->inputs_.get("x"), CUDNN_TENSOR_NCHW);
    auto dx = x->makeSharedGrad();
    auto y = n->outputs_.get("y");

    auto yl = std::make_shared<CudaTensor>(
        x->m_storage, y->dims_.n(p.m_batch_size), CUDNN_TENSOR_NCHW,
        x->namePostfix("reshape"));

    p.m_tensors[y] = yl;
    yl->m_grad = std::make_shared<CudaTensor>(
        dx->m_storage, y->dims_.n(p.m_batch_size), CUDNN_TENSOR_NCHW,
        x->namePostfix("reshape"));
    return {};
}

static Nodes
reshape_transform(CudaProgram &p, const Nodes &nodes)
{
    Nodes r;

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
window_transform_node(CudaProgram &p, const Node &n)
{
    auto x = p.lower_tensor(n.inputs_.get("x"), CUDNN_TENSOR_NCHW);
    auto dx = x->makeSharedGrad();
    auto y = n.outputs_.get("y");

    Dims offset(n.attributes_.get("offset", std::vector<int>{}));

    auto yl = std::make_shared<CudaTensor>(
        x, y->dims_.n(p.m_batch_size), offset.i64(), y->namePostfix("alias"));

    p.m_tensors[y] = yl;

    yl->m_grad = std::make_shared<CudaTensor>(
        dx, y->dims_.n(p.m_batch_size), offset.i64(), y->namePostfix("alias"));
}

static Nodes
window_transform(CudaProgram &p, const Nodes &nodes)
{
    Nodes r;

    for(size_t i = 0; i < nodes.size(); i++) {
        auto &n = nodes[i];
        if(n->type_ == "window") {
            window_transform_node(p, *n);
        } else {
            r.push_back(n);
        }
    }
    return r;
}

REGISTER_CUDA_TRANSFORM(100, CUDA_TRANSFORM_ALL, window_transform);

//------------------------------------------------------------------------

static void
concat_transform_node(CudaProgram &p, const Node &n)
{
    const int axis = n.attributes_.get("axis", 1);

    auto y = p.lower_tensor(n.outputs_.get("y"));
    auto dy = y->makeSharedGrad();
    auto element_offset = std::vector<int64_t>(y->dims_.size(), 0);

    for(const auto &xh : n.inputs_.getv("x")) {
        auto x = std::make_shared<CudaTensor>(y, xh->dims_.n(p.m_batch_size),
                                              element_offset,
                                              xh->namePostfix("alias"));
        x->copyFromLocked(*xh);
        p.m_tensors[xh] = x;
        x->m_grad = std::make_shared<CudaTensor>(
            dy, xh->dims_.n(p.m_batch_size), element_offset,
            xh->namePostfix("alias"));
        element_offset[axis] += xh->dims_[axis];
    }
}

static Nodes
concat_transform(CudaProgram &p, const Nodes &nodes)
{
    Nodes r;

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

struct CudaStats : public CudaOperation {
    const std::shared_ptr<CudaTensor> x_, y_;
    const size_t m_elements;
    CudaStats(std::shared_ptr<CudaTensor> x, std::shared_ptr<CudaTensor> y)
      : CudaOperation("stats"), x_(x), y_(y), m_elements(x->dims_.elements())
    {
    }

    const char *exec(CudaProgram &p, long batch)
    {
        switch(x_->data_type_) {
        case Tensor::DataType::FLOAT:
            tensor_stats_float(m_elements, (const float *)x_->deviceMem(),
                               (float *)y_->deviceMem(), p.m_ctx->m_stream);
            break;
        case Tensor::DataType::HALF:
            tensor_stats_half(m_elements, (const __half *)x_->deviceMem(),
                              (float *)y_->deviceMem(), p.m_ctx->m_stream);
            break;
        default:
            return "Unsupported datatype";
        }
        return NULL;
    }

    std::vector<std::shared_ptr<CudaTensor>> getInputs() const override
    {
        return {x_};
    }

    std::vector<std::shared_ptr<CudaTensor>> getOutputs() const override
    {
        return {y_};
    }
};

static const char *
stats_setup(CudaProgram &p, const Node &n, bool training)
{
    auto it = p.m_tensors.find(n.inputs_.get("x"));
    if(it == p.m_tensors.end())
        return "x-tensor not found";

    auto x = it->second;
    if(n.attributes_.get("gradient", false)) {
        x = x->m_grad;

        if(!x)
            return "x-tensor has no gradient";
    }

    auto y = p.lower_tensor(n.outputs_.get("y"));

    auto op = std::make_shared<CudaStats>(x, y);

    if(training) {
        p.upd(op);
    } else {
        p.infer(op);
    }
    return NULL;
}

REGISTER_CUDA_OP("stats", stats_setup);

}  // namespace saga
