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

#include "cuda_common.hpp"
#include "cuda_tensor.hpp"
#include "cuda_kernels.hpp"

namespace saga {

//------------------------------------------------------------------------

struct CudnnAddTensor : public CudaOperation {
    const std::shared_ptr<CudaTensor> x_, y_;
    float m_beta{1.0f};

    CudnnAddTensor(std::shared_ptr<CudaTensor> x, std::shared_ptr<CudaTensor> y)
      : CudaOperation("add"), x_(x), y_(y)
    {
    }

    void exec(CudaProgram &p) override
    {
        float alpha = 1.0f;
        assert(m_beta == 1.0f);

        chkCUDNN(cudnnAddTensor(p.m_ctx->m_cudnn, &alpha, x_->desc(),
                                x_->deviceMem(), &alpha, y_->desc(),
                                y_->deviceMem()));
    }

    CudaOpArgs listInputs() const override { return {{"x", x_}}; }

    CudaOpArgs listOutputs() const override { return {{"y", y_}}; }

    float *getBeta(const std::shared_ptr<CudaTensor> &t) override
    {
        if(t == y_)
            return &m_beta;
        return nullptr;
    }
};

//------------------------------------------------------------------------

struct CudnnTransform : public CudaOperation {
    const std::shared_ptr<CudaTensor> a_, b_;
    float m_beta{0};

    CudnnTransform(std::shared_ptr<CudaTensor> a, std::shared_ptr<CudaTensor> b)
      : CudaOperation("transform"), a_(a), b_(b)
    {
    }

    void exec(CudaProgram &p) override
    {
        float alpha = 1.0f;
        chkCUDNN(cudnnTransformTensor(p.m_ctx->m_cudnn, &alpha, a_->desc(),
                                      a_->deviceMem(), &m_beta, b_->desc(),
                                      b_->deviceMem()));
    }

    CudaOpArgs listInputs() const override { return {{"a", a_}}; }

    CudaOpArgs listOutputs() const override { return {{"b", b_}}; }

    float *getBeta(const std::shared_ptr<CudaTensor> &t) override
    {
        if(t == b_)
            return &m_beta;
        return nullptr;
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

    void setup(CudaProgram &p, CudaTensor &x, CudaTensor &w, CudaTensor &y,
               int pad, int stride, int group, bool bwd)
    {
        auto algo_key = cudnn_conv_hash_key(x, w, y, pad, stride);

        auto wfmt = y.format();
        auto dima = w.m_dims.i32();
        chkCUDNN(cudnnSetFilterNdDescriptor(filter_desc_, x.m_type, wfmt,
                                            dima.size(), dima.data()));

        chkCUDNN(cudnnSetConvolutionMathType(conv_desc_, CUDNN_TENSOR_OP_MATH));

        chkCUDNN(cudnnSetConvolutionGroupCount(conv_desc_, group));

        size_t spatial_dims = x.m_dims.size() - 2;

        assert(spatial_dims <= 3);
        int padA[3];
        int strideA[3];
        int dilationA[3];

        for(size_t i = 0; i < spatial_dims; i++) {
            padA[i] = pad;
            strideA[i] = stride;
            dilationA[i] = 1;
        }

        chkCUDNN(cudnnSetConvolutionNdDescriptor(
            conv_desc_, spatial_dims, padA, strideA, dilationA,
            CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT))

            auto fwd_algo_key = std::string("conv.fwd;") + algo_key;
        if(auto it = p.m_ctx->m_algo_hash.find(fwd_algo_key);
           it != p.m_ctx->m_algo_hash.end()) {
            conv_fwd_algo_ = (cudnnConvolutionFwdAlgo_t)it->second;
        } else {
            int count;
            chkCUDNN(cudnnGetConvolutionForwardAlgorithmMaxCount(
                p.m_ctx->m_cudnn, &count))

                cudnnConvolutionFwdAlgoPerf_t fwdalgos[count];

            chkCUDNN(cudnnFindConvolutionForwardAlgorithm(
                p.m_ctx->m_cudnn, x.m_desc, filter_desc_, conv_desc_, y.m_desc,
                count, &count, fwdalgos));

            if(count == 0)
                throw std::runtime_error("No forwarding algo found");

            conv_fwd_algo_ = fwdalgos[0].algo;
            p.m_ctx->m_algo_hash[fwd_algo_key] = conv_fwd_algo_;
        }
        size_t workspace;
        chkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(
            p.m_ctx->m_cudnn, x.m_desc, filter_desc_, conv_desc_, y.m_desc,
            conv_fwd_algo_, &workspace));

        p.m_ctx->m_workspace.request(workspace);

        if(!bwd)
            return;

        auto bwd_data_algo_key = std::string("conv.bwd.data;") + algo_key;
        if(auto it = p.m_ctx->m_algo_hash.find(bwd_data_algo_key);
           it != p.m_ctx->m_algo_hash.end()) {
            bwd_data_algo_ = (cudnnConvolutionBwdDataAlgo_t)it->second;
        } else {
            int count;
            chkCUDNN(cudnnGetConvolutionBackwardDataAlgorithmMaxCount(
                p.m_ctx->m_cudnn, &count))

                cudnnConvolutionBwdDataAlgoPerf_t bwdalgos[count];

            chkCUDNN(cudnnFindConvolutionBackwardDataAlgorithm(
                p.m_ctx->m_cudnn, filter_desc_, y.desc(), conv_desc_, x.desc(),
                count, &count, bwdalgos));

            if(count == 0)
                throw std::runtime_error("No backward data algo found");

            bwd_data_algo_ = bwdalgos[0].algo;
            p.m_ctx->m_algo_hash[bwd_data_algo_key] = bwd_data_algo_;
        }
        chkCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(
            p.m_ctx->m_cudnn, filter_desc_, y.desc(), conv_desc_, x.desc(),
            bwd_data_algo_, &workspace));

        p.m_ctx->m_workspace.request(workspace);

        auto bwd_filter_algo_key = std::string("conv.bwd.filter;") + algo_key;
        if(auto it = p.m_ctx->m_algo_hash.find(bwd_filter_algo_key);
           it != p.m_ctx->m_algo_hash.end()) {
            bwd_filter_algo_ = (cudnnConvolutionBwdFilterAlgo_t)it->second;
        } else {
            int count;
            chkCUDNN(cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(
                p.m_ctx->m_cudnn, &count));
            cudnnConvolutionBwdFilterAlgoPerf_t filteralgos[count];

            chkCUDNN(cudnnFindConvolutionBackwardFilterAlgorithm(
                p.m_ctx->m_cudnn, x.desc(), y.desc(), conv_desc_, filter_desc_,
                count, &count, filteralgos));

            if(count == 0)
                throw std::runtime_error("No backward filer algo found");

            bwd_filter_algo_ = filteralgos[0].algo;
            p.m_ctx->m_algo_hash[bwd_filter_algo_key] = bwd_filter_algo_;
        }
        chkCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(
            p.m_ctx->m_cudnn, x.desc(), y.desc(), conv_desc_, filter_desc_,
            bwd_filter_algo_, &workspace));

        p.m_ctx->m_workspace.request(workspace);
    }
};

struct CudnnConvolutionFwd : public CudaOperation {
    const std::shared_ptr<CudaContext> ctx_;
    const std::shared_ptr<CudnnConvolutionDesc> desc_;
    const std::shared_ptr<CudaTensor> x_, w_, y_;
    float m_beta{0};

    CudnnConvolutionFwd(std::shared_ptr<CudaContext> ctx,
                        std::shared_ptr<CudnnConvolutionDesc> desc,
                        std::shared_ptr<CudaTensor> x,
                        std::shared_ptr<CudaTensor> w,
                        std::shared_ptr<CudaTensor> y)
      : CudaOperation("convfwd"), ctx_(ctx), desc_(desc), x_(x), w_(w), y_(y)
    {
    }

    void exec(CudaProgram &p) override
    {
        float alpha = 1.0f;

        chkCUDNN(cudnnConvolutionForward(
            ctx_->m_cudnn, &alpha, x_->desc(), x_->deviceMem(),
            desc_->filter_desc_, w_->deviceMem(), desc_->conv_desc_,
            desc_->conv_fwd_algo_, ctx_->m_workspace.ptr(),
            ctx_->m_workspace.size(), &m_beta, y_->desc(), y_->deviceMem()));
    }

    CudaOpArgs listInputs() const override { return {{"x", x_}, {"w", w_}}; }

    CudaOpArgs listOutputs() const override { return {{"y", y_}}; }

    std::string info() const override
    {
        return convfwdalgostr(desc_->conv_fwd_algo_);
    }

    float *getBeta(const std::shared_ptr<CudaTensor> &t) override
    {
        if(t == y_)
            return &m_beta;
        return nullptr;
    }
};

struct CudnnConvolutionBwdBias : public CudaOperation {
    const std::shared_ptr<CudaContext> ctx_;
    const std::shared_ptr<CudaTensor> dy_, db_;
    float m_beta{0};

    CudnnConvolutionBwdBias(std::shared_ptr<CudaContext> ctx,
                            std::shared_ptr<CudaTensor> dy,
                            std::shared_ptr<CudaTensor> db)
      : CudaOperation("convbwdbias"), ctx_(ctx), dy_(dy), db_(db)
    {
    }

    void exec(CudaProgram &p) override
    {
        float alpha = 1.0f;

        chkCUDNN(cudnnConvolutionBackwardBias(
            ctx_->m_cudnn, &alpha, dy_->desc(), dy_->deviceMem(), &m_beta,
            db_->desc(), db_->deviceMem()));
    }

    CudaOpArgs listInputs() const override { return {{"dy", dy_}}; }

    CudaOpArgs listOutputs() const override { return {{"db", db_}}; }

    float *getBeta(const std::shared_ptr<CudaTensor> &t) override
    {
        if(t == db_)
            return &m_beta;
        return nullptr;
    }
};

struct CudnnConvolutionBwdFilter : public CudaOperation {
    const std::shared_ptr<CudaContext> ctx_;
    const std::shared_ptr<CudnnConvolutionDesc> desc_;
    const std::shared_ptr<CudaTensor> x_, dy_, dw_;
    float m_beta{0};

    CudnnConvolutionBwdFilter(std::shared_ptr<CudaContext> ctx,
                              std::shared_ptr<CudnnConvolutionDesc> desc,
                              std::shared_ptr<CudaTensor> x,
                              std::shared_ptr<CudaTensor> dy,
                              std::shared_ptr<CudaTensor> dw)
      : CudaOperation("convbwdfilter")
      , ctx_(ctx)
      , desc_(desc)
      , x_(x)
      , dy_(dy)
      , dw_(dw)
    {
    }

    void exec(CudaProgram &p) override
    {
        float alpha = 1.0f;

        chkCUDNN(cudnnConvolutionBackwardFilter(
            ctx_->m_cudnn, &alpha, x_->desc(), x_->deviceMem(), dy_->desc(),
            dy_->deviceMem(), desc_->conv_desc_, desc_->bwd_filter_algo_,
            ctx_->m_workspace.ptr(), ctx_->m_workspace.size(), &m_beta,
            desc_->filter_desc_, dw_->deviceMem()));
    }

    CudaOpArgs listInputs() const override { return {{"x", x_}, {"dy", dy_}}; }

    CudaOpArgs listOutputs() const override { return {{"dw", dw_}}; }

    std::string info() const override
    {
        return convbwdfilteralgostr(desc_->bwd_filter_algo_);
    }

    float *getBeta(const std::shared_ptr<CudaTensor> &t) override
    {
        if(t == dw_)
            return &m_beta;
        return nullptr;
    }
};

struct CudnnConvolutionBwdData : public CudaOperation {
    const std::shared_ptr<CudaContext> ctx_;
    const std::shared_ptr<CudnnConvolutionDesc> desc_;
    const std::shared_ptr<CudaTensor> w_, dy_, dx_;
    float m_beta{0};

    CudnnConvolutionBwdData(std::shared_ptr<CudaContext> ctx,
                            std::shared_ptr<CudnnConvolutionDesc> desc,
                            std::shared_ptr<CudaTensor> w,
                            std::shared_ptr<CudaTensor> dy,
                            std::shared_ptr<CudaTensor> dx)
      : CudaOperation("convbwddata")
      , ctx_(ctx)
      , desc_(desc)
      , w_(w)
      , dy_(dy)
      , dx_(dx)
    {
    }

    void exec(CudaProgram &p) override
    {
        float alpha = 1.0f;

        chkCUDNN(cudnnConvolutionBackwardData(
            ctx_->m_cudnn, &alpha, desc_->filter_desc_, w_->deviceMem(),
            dy_->desc(), dy_->deviceMem(), desc_->conv_desc_,
            desc_->bwd_data_algo_, ctx_->m_workspace.ptr(),
            ctx_->m_workspace.size(), &m_beta, dx_->desc(), dx_->deviceMem()));
    }

    CudaOpArgs listInputs() const override { return {{"w", w_}, {"dy", dy_}}; }

    CudaOpArgs listOutputs() const override { return {{"dx", dx_}}; }

    std::string info() const override
    {
        return convbwddataalgostr(desc_->bwd_data_algo_);
    }

    float *getBeta(const std::shared_ptr<CudaTensor> &t) override
    {
        if(t == dx_)
            return &m_beta;
        return nullptr;
    }
};

static const void
conv_setup(CudaProgram &p, CudaProgramUnit &pu, const Node &n)
{
    auto x = p.lower_tensor(pu, n.m_inputs["x"]);
    auto y = p.lower_tensor(pu, n.m_outputs["y"]);
    auto w = p.lower_tensor(pu, n.m_inputs["w"], y->format());
    auto b = n.m_inputs.has("b")
                 ? p.lower_tensor(pu, n.m_inputs["b"], y->m_dims.size())
                 : nullptr;

    auto desc = std::make_shared<CudnnConvolutionDesc>();

    const int pad = n.m_attributes.get("pad", 0);
    const int stride = n.m_attributes.get("stride", 1);
    const bool transpose = n.m_attributes.get("transpose", false);
    const int group = n.m_attributes.get("group", 1);
    if(!transpose) {
        // Non-transposed (Standard convolution)

        desc->setup(p, *x, *w, *y, pad, stride, group,
                    p.m_pt == ProgramType::TRAINING);

        pu.fwd(std::make_shared<CudnnConvolutionFwd>(p.m_ctx, desc, x, w, y));
        if(b)
            pu.fwd(std::make_shared<CudnnAddTensor>(b, y));

        if(p.m_pt == ProgramType::INFERENCE)
            return;

        auto dy = p.lower_grad(pu, n.m_outputs["y"]);
        if(b) {
            auto db = p.lower_grad(pu, n.m_inputs["b"], y->m_dims.size());
            pu.bwd(std::make_shared<CudnnConvolutionBwdBias>(p.m_ctx, dy, db));
            p.upd(n.m_inputs["b"], b, db);
        }

        auto dw = p.lower_grad(pu, n.m_inputs["w"], y->format());
        pu.bwd(std::make_shared<CudnnConvolutionBwdFilter>(p.m_ctx, desc, x, dy,
                                                           dw));
        p.upd(n.m_inputs["w"], w, dw);

        auto dx = p.lower_grad(pu, n.m_inputs["x"]);
        if(dx) {
            pu.bwd(std::make_shared<CudnnConvolutionBwdData>(p.m_ctx, desc, w,
                                                             dy, dx));
        }

    } else {
        // Transposed ("up-convolution" / "fractionally strided convolution")

        desc->setup(p, *y, *w, *x, pad, stride, group, true);

        pu.fwd(
            std::make_shared<CudnnConvolutionBwdData>(p.m_ctx, desc, w, x, y));
        if(b)
            pu.fwd(std::make_shared<CudnnAddTensor>(b, y));

        if(p.m_pt == ProgramType::INFERENCE)
            return;

        auto dy = p.lower_grad(pu, n.m_outputs["y"]);
        if(b) {
            auto db = p.lower_grad(pu, n.m_inputs["b"], y->m_dims.size());
            pu.bwd(std::make_shared<CudnnConvolutionBwdBias>(p.m_ctx, dy, db));
            p.upd(n.m_inputs["b"], b, db);
        }

        // Update weights
        auto dw = p.lower_grad(pu, n.m_inputs["w"], y->format());
        pu.bwd(std::make_shared<CudnnConvolutionBwdFilter>(p.m_ctx, desc, dy, x,
                                                           dw));
        p.upd(n.m_inputs["w"], w, dw);

        // Backprop
        auto dx = p.lower_grad(pu, n.m_inputs["x"]);
        if(dx) {
            pu.bwd(std::make_shared<CudnnConvolutionFwd>(p.m_ctx, desc, dy, w,
                                                         dx));
        }
    }
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

struct CudnnActivationFwd : public CudaOperation {
    const std::shared_ptr<CudaTensor> x_, y_;
    float m_beta{0};

    cudnnActivationDescriptor_t desc_;

    CudnnActivationFwd(std::shared_ptr<CudaTensor> x,
                       std::shared_ptr<CudaTensor> y,
                       cudnnActivationMode_t mode, float alpha)
      : CudaOperation("actfwd"), x_(x), y_(y)
    {
        chkCUDNN(cudnnCreateActivationDescriptor(&desc_));
        chkCUDNN(cudnnSetActivationDescriptor(desc_, mode, CUDNN_PROPAGATE_NAN,
                                              alpha));
        if(mode == CUDNN_ACTIVATION_SWISH) {
            chkCUDNN(cudnnSetActivationDescriptorSwishBeta(desc_, 1.0));
        }
    }

    ~CudnnActivationFwd() { chkCUDNN(cudnnDestroyActivationDescriptor(desc_)); }

    void exec(CudaProgram &p) override
    {
        float alpha = 1.0f;

        chkCUDNN(cudnnActivationForward(p.m_ctx->m_cudnn, desc_, &alpha,
                                        x_->desc(), x_->deviceMem(), &m_beta,
                                        y_->desc(), y_->deviceMem()));
    }

    CudaOpArgs listInputs() const override { return {{"x", x_}}; }

    CudaOpArgs listOutputs() const override { return {{"y", y_}}; }

    std::string info() const override { return activationalgo_to_str(desc_); }

    float *getBeta(const std::shared_ptr<CudaTensor> &t) override
    {
        if(t == y_)
            return &m_beta;
        return nullptr;
    }
};

struct CudnnActivationBwd : public CudaOperation {
    const std::shared_ptr<CudnnActivationFwd> fwd_;
    const std::shared_ptr<CudaTensor> x_, y_, dx_, dy_;
    float m_beta{0};

    CudnnActivationBwd(const std::shared_ptr<CudnnActivationFwd> fwd,
                       std::shared_ptr<CudaTensor> x,
                       std::shared_ptr<CudaTensor> y,
                       std::shared_ptr<CudaTensor> dx,
                       std::shared_ptr<CudaTensor> dy)
      : CudaOperation("actbwd"), fwd_(fwd), x_(x), y_(y), dx_(dx), dy_(dy)
    {
    }

    void exec(CudaProgram &p) override
    {
        float alpha = 1.0f;

        chkCUDNN(cudnnActivationBackward(
            p.m_ctx->m_cudnn, fwd_->desc_, &alpha, y_->desc(), y_->deviceMem(),
            dy_->desc(), dy_->deviceMem(), x_->desc(), x_->deviceMem(), &m_beta,
            dx_->desc(), dx_->deviceMem()));
    }

    CudaOpArgs listInputs() const override
    {
        return {{"x", x_}, {"y", y_}, {"dy", dy_}};
    }

    CudaOpArgs listOutputs() const override { return {{"dx", dx_}}; }

    std::string info() const override
    {
        return activationalgo_to_str(fwd_->desc_);
    }

    float *getBeta(const std::shared_ptr<CudaTensor> &t) override
    {
        if(t == dx_)
            return &m_beta;
        return nullptr;
    }
};

static void
activation_setup(CudaProgram &p, CudaProgramUnit &pu, const Node &n,
                 cudnnActivationMode_t mode, float alpha)
{
    auto x = p.lower_tensor(pu, n.m_inputs["x"]);
    auto y = p.lower_tensor(pu, n.m_outputs["y"]);
    auto fwd = std::make_shared<CudnnActivationFwd>(x, y, mode, alpha);

    pu.fwd(fwd);

    if(p.m_pt == ProgramType::INFERENCE)
        return;

    auto dx = p.lower_grad(pu, n.m_inputs["x"]);
    auto dy = p.lower_grad(pu, n.m_outputs["y"]);

    pu.bwd(std::make_shared<CudnnActivationBwd>(fwd, x, y, dx, dy));
}

static void
relu_setup(CudaProgram &p, CudaProgramUnit &pu, const Node &n)
{
    activation_setup(p, pu, n, CUDNN_ACTIVATION_RELU, 0.0f);
}

REGISTER_CUDA_OP("relu", relu_setup);

static void
elu_setup(CudaProgram &p, CudaProgramUnit &pu, const Node &n)
{
    activation_setup(p, pu, n, CUDNN_ACTIVATION_ELU,
                     n.m_attributes.get("alpha", 0.1f));
}

REGISTER_CUDA_OP("elu", relu_setup);

static void
sigmoid_setup(CudaProgram &p, CudaProgramUnit &pu, const Node &n)
{
    activation_setup(p, pu, n, CUDNN_ACTIVATION_SIGMOID, 0.0f);
}

REGISTER_CUDA_OP("sigmoid", sigmoid_setup);

static void
tanh_setup(CudaProgram &p, CudaProgramUnit &pu, const Node &n)
{
    activation_setup(p, pu, n, CUDNN_ACTIVATION_TANH, 0.0f);
}

REGISTER_CUDA_OP("tanh", tanh_setup);

static void
swish_setup(CudaProgram &p, CudaProgramUnit &pu, const Node &n)
{
    activation_setup(p, pu, n, CUDNN_ACTIVATION_SWISH, 0.0f);
}

REGISTER_CUDA_OP("swish", swish_setup);

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
      , m_elements(x->m_dims.elements())
    {
    }

    void exec(CudaProgram &p) override
    {
        switch(x_->m_data_type) {
        case Tensor::DataType::FLOAT:
            leaky_relu_float(m_elements, (float *)y_->deviceMem(),
                             (const float *)x_->deviceMem(), alpha_,
                             p.m_ctx->m_stream);
            break;
        default:
            throw std::runtime_error{"Unsupported datatype"};
        }
    }

    CudaOpArgs listInputs() const override { return {{"x", x_}}; }

    CudaOpArgs listOutputs() const override { return {{"y", y_}}; }
};

static void
leakyrelu_setup(CudaProgram &p, CudaProgramUnit &pu, const Node &n)
{
    if(p.m_pt != ProgramType::INFERENCE)
        throw std::runtime_error{"LeakRelu not supported for training"};

    auto x = p.lower_tensor(pu, n.m_inputs["x"]);
    auto y = p.lower_tensor(pu, n.m_outputs["y"]);
    float alpha = n.m_attributes.get("alpha", 0.01f);
    auto op = std::make_shared<CudaLeakyRelu>(x, y, alpha);
    pu.fwd(op);
}

REGISTER_CUDA_OP("leakyrelu", leakyrelu_setup);

//------------------------------------------------------------------------

struct CudnnPoolingFwd : public CudaOperation {
    const std::shared_ptr<CudaTensor> x_, y_;

    cudnnPoolingDescriptor_t desc_;

    CudnnPoolingFwd(std::shared_ptr<CudaTensor> x,
                    std::shared_ptr<CudaTensor> y, cudnnPoolingMode_t mode,
                    int size, int pad, int stride)
      : CudaOperation("poolfwd"), x_(x), y_(y)
    {
        chkCUDNN(cudnnCreatePoolingDescriptor(&desc_));

        chkCUDNN(cudnnSetPooling2dDescriptor(desc_, mode, CUDNN_PROPAGATE_NAN,
                                             size, size, pad, pad, stride,
                                             stride));
    }

    ~CudnnPoolingFwd() { chkCUDNN(cudnnDestroyPoolingDescriptor(desc_)); }

    void exec(CudaProgram &p) override
    {
        float alpha = 1.0f;
        float beta = 0.0f;

        chkCUDNN(cudnnPoolingForward(p.m_ctx->m_cudnn, desc_, &alpha,
                                     x_->desc(), x_->deviceMem(), &beta,
                                     y_->desc(), y_->deviceMem()));
    }

    CudaOpArgs listInputs() const override { return {{"x", x_}}; }

    CudaOpArgs listOutputs() const override { return {{"y", y_}}; }
};

struct CudnnPoolingBwd : public CudaOperation {
    const std::shared_ptr<CudnnPoolingFwd> fwd_;
    const std::shared_ptr<CudaTensor> x_, y_, dx_, dy_;
    float m_beta{0};

    CudnnPoolingBwd(std::shared_ptr<CudnnPoolingFwd> fwd,
                    std::shared_ptr<CudaTensor> x,
                    std::shared_ptr<CudaTensor> y,
                    std::shared_ptr<CudaTensor> dx,
                    std::shared_ptr<CudaTensor> dy)
      : CudaOperation("poolbwd"), fwd_(fwd), x_(x), y_(y), dx_(dx), dy_(dy)
    {
    }

    void exec(CudaProgram &p) override
    {
        float alpha = 1.0f;

        chkCUDNN(cudnnPoolingBackward(
            p.m_ctx->m_cudnn, fwd_->desc_, &alpha, y_->desc(), y_->deviceMem(),
            dy_->desc(), dy_->deviceMem(), x_->desc(), x_->deviceMem(), &m_beta,
            dx_->desc(), dx_->deviceMem()));
    }

    CudaOpArgs listInputs() const override
    {
        return {{"x", x_}, {"y", y_}, {"dy", dy_}};
    }

    CudaOpArgs listOutputs() const override { return {{"dx", dx_}}; }

    float *getBeta(const std::shared_ptr<CudaTensor> &t) override
    {
        if(t == dx_)
            return &m_beta;
        return nullptr;
    }
};

static void
pooling_setup(CudaProgram &p, CudaProgramUnit &pu, const Node &n,
              cudnnPoolingMode_t mode)
{
    auto x = p.lower_tensor(pu, n.m_inputs["x"]);
    auto y = p.lower_tensor(pu, n.m_outputs["y"]);

    int size;
    if(n.m_attributes.get("global", false)) {
        size = x->m_dims[2];
    } else {
        size = n.m_attributes.get("size", 1);
    }
    const int pad = n.m_attributes.get("pad", 0);
    const int stride = n.m_attributes.get("stride", 1);

    auto fwd = std::make_shared<CudnnPoolingFwd>(x, y, mode, size, pad, stride);

    pu.fwd(fwd);

    if(p.m_pt == ProgramType::INFERENCE)
        return;

    auto dx = p.lower_grad(pu, n.m_inputs["x"]);
    auto dy = p.lower_grad(pu, n.m_outputs["y"]);

    pu.bwd(std::make_shared<CudnnPoolingBwd>(fwd, x, y, dx, dy));
}

static void
maxpool_setup(CudaProgram &p, CudaProgramUnit &pu, const Node &n)
{
    pooling_setup(p, pu, n, CUDNN_POOLING_MAX);
}

REGISTER_CUDA_OP("maxpool", maxpool_setup);

static void
avgpool_setup(CudaProgram &p, CudaProgramUnit &pu, const Node &n)
{
    pooling_setup(p, pu, n, CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING);
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
    float m_beta{0};

    CudaGemm(cublasOperation_t transa, cublasOperation_t transb, int m, int n,
             int k, std::shared_ptr<CudaTensor> a, int lda,
             std::shared_ptr<CudaTensor> b, int ldb,
             std::shared_ptr<CudaTensor> c, int ldc, const char *name)
      : CudaOperation(name)
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

    void exec(CudaProgram &p) override
    {
        float alpha = 1.0f, beta = m_beta;
        __half halpha = 1.0f, hbeta = m_beta;

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
            throw std::runtime_error{"Unsupported tensor datatype"};
        }
        if(s)
            throw std::runtime_error{cublasErrStr(s)};
    }

    CudaOpArgs listInputs() const override { return {{"a", a_}, {"b", b_}}; }

    CudaOpArgs listOutputs() const override { return {{"c", c_}}; }

    float *getBeta(const std::shared_ptr<CudaTensor> &t) override
    {
        if(t == c_)
            return &m_beta;
        return nullptr;
    }
};

/**
 * https://docs.nvidia.com/deeplearning/performance/dl-performance-fully-connected/index.html#fullyconnected-layer
 */

static void
fc_setup(CudaProgram &p, CudaProgramUnit &pu, const Node &n)
{
    auto x = p.lower_tensor(pu, n.m_inputs["x"]);
    auto y = p.lower_tensor(pu, n.m_outputs["y"]);
    auto w = p.lower_tensor(pu, n.m_inputs["w"]);
    auto b =
        n.m_inputs.has("b") ? p.lower_tensor(pu, n.m_inputs["b"], 2) : nullptr;

    const bool transW = n.m_attributes.get("transW", false);
    const int num_inputs = x->m_dims[1];
    const int num_outputs = y->m_dims[1];
    const int batch_size = x->m_dims[0];

    // clang-format off
    auto fwd = std::make_shared<CudaGemm>(
        transW ? CUBLAS_OP_T : CUBLAS_OP_N,
        CUBLAS_OP_N,
        num_outputs, batch_size, num_inputs,
        w, transW ? num_inputs : num_outputs,
        x, num_inputs,
        y, num_outputs,
        transW ? "fc.fwd.t" : "fc.fwd.n");
    // clang-format on

    pu.fwd(fwd);

    if(b)
        pu.fwd(std::make_shared<CudnnAddTensor>(b, y));

    if(p.m_pt == ProgramType::INFERENCE)
        return;

    auto dx = p.lower_grad(pu, n.m_inputs["x"]);
    auto dy = p.lower_grad(pu, n.m_outputs["y"]);
    auto dw = p.lower_grad(pu, n.m_inputs["w"]);

    // clang-format off
    if(transW) {
        pu.bwd(std::make_shared<CudaGemm>(
                   CUBLAS_OP_N,
                   CUBLAS_OP_T,
                   num_inputs, num_outputs, batch_size,
                   x, num_inputs,
                   dy, num_outputs,
                   dw, num_inputs,
                   "fc.bwd.weights.t"));
    } else {
        pu.bwd(std::make_shared<CudaGemm>(
                   CUBLAS_OP_N,
                   CUBLAS_OP_T,
                   num_outputs, num_inputs, batch_size,
                   dy, num_outputs,
                   x, num_inputs,
                   dw, num_outputs,
                   "fc.bwd.weights.n"));
    }
    // clang-format on

    p.upd(n.m_inputs["w"], w, dw);

    if(b) {
        auto ones = p.lower_tensor(
            pu, Tensor::make(x->m_data_type, {batch_size, 1}, 1, 0));

        auto db = p.lower_grad(pu, n.m_inputs["b"], 2);
        pu.bwd(std::make_shared<CudaGemm>(CUBLAS_OP_N, CUBLAS_OP_T, 1,
                                          num_outputs, batch_size, ones, 1, dy,
                                          num_outputs, db, 1, "fc.bwd.bias"));

        p.upd(n.m_inputs["b"], b, db);
    }

    if(dx) {
        // clang-format off
        pu.bwd(std::make_shared<CudaGemm>(
            transW ? CUBLAS_OP_N : CUBLAS_OP_T,
            CUBLAS_OP_N,
            num_inputs, batch_size, num_outputs,
            w, transW ? num_inputs : num_outputs,
            dy, num_outputs,
            dx, num_inputs,
            transW ? "fc.bwd.data.t" : "fc.bwd.data.n"));
        // clang-format on
    }
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
      , m_elements(x->m_dims.elements())
    {
        if(x_->m_data_type == Tensor::DataType::U8 &&
           y_->m_data_type == Tensor::DataType::FLOAT) {
            algo_ = convert_u8_float;
        } else if(x_->m_data_type == Tensor::DataType::U8 &&
                  y_->m_data_type == Tensor::DataType::HALF) {
            algo_ = convert_u8_half;
        } else if(x_->m_data_type == Tensor::DataType::FLOAT &&
                  y_->m_data_type == Tensor::DataType::HALF) {
            algo_ = convert_float_half;
        } else if(x_->m_data_type == Tensor::DataType::I16 &&
                  y_->m_data_type == Tensor::DataType::HALF) {
            algo_ = convert_i16_half;
        } else {
            throw std::runtime_error{fmt("Unable to convert %s -> %s",
                                         Tensor::DataTypeStr(x_->m_data_type),
                                         Tensor::DataTypeStr(y_->m_data_type))};
        }
    }

    void exec(CudaProgram &p) override
    {
        algo_(x_->deviceMem(), y_->deviceMem(), m_elements, scale_,
              p.m_ctx->m_stream);
    }

    CudaOpArgs listInputs() const override { return {{"x", x_}}; }

    CudaOpArgs listOutputs() const override { return {{"y", y_}}; }
};

static void
convert_setup(CudaProgram &p, CudaProgramUnit &pu, const Node &n)
{
    auto scale = n.m_attributes.get("scale", 1.0f);

    auto xh = n.m_inputs["x"];
    auto x = p.lower_tensor(pu, xh);

    auto yh = n.m_outputs["y"];

    if(xh->m_data_type == yh->m_data_type && scale == 1.0f) {
        auto y =
            std::make_shared<CudaTensor>(x, x->m_dims, std::vector<int64_t>{},
                                         xh->namePostfix("nop-convert"));
        p.m_ctx->m_tensors[yh] = y;
        return;
    }

    auto y = p.lower_tensor(yh, *x);
    pu.fwd(std::make_shared<CudaConvert>(x, y, scale));
}

REGISTER_CUDA_OP("convert", convert_setup);

static std::shared_ptr<Node>
convert_fuse_nodes(CudaProgram &p, std::shared_ptr<Node> a,
                   std::shared_ptr<Node> b)
{
    float scale =
        a->m_attributes.get("scale", 1.0f) * b->m_attributes.get("scale", 1.0f);

    auto nn = std::make_shared<Node>("convert");
    nn->m_inputs["x"] = a->m_inputs["x"];
    nn->m_outputs["y"] = b->m_outputs["y"];
    nn->m_attributes["scale"] = scale;
    return nn;
}

static Nodes
convert_transform(CudaProgram &p, CudaProgramUnit &pu, const Nodes &input)
{
    Nodes nodes = input;

again:

    for(auto &n : nodes) {
        if(n->m_type != "convert")
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

REGISTER_CUDA_TRANSFORM(200, ProgramType::INFERENCE | ProgramType::TRAINING,
                        convert_transform);

//------------------------------------------------------------------------
struct CudaCatClassifierFwd : public CudaOperation {
    const std::shared_ptr<CudaTensor> x_, y_;

    CudaCatClassifierFwd(std::shared_ptr<CudaTensor> x,
                         std::shared_ptr<CudaTensor> y)
      : CudaOperation("catclassifierfwd"), x_(x), y_(y)
    {
    }

    void exec(CudaProgram &p) override
    {
        switch(x_->m_type) {
        case CUDNN_DATA_FLOAT:
            catclassifier_fwd_float_i32(
                x_->m_dims[0], (const float *)x_->deviceMem(),
                (int32_t *)y_->deviceMem(), x_->m_dims[1], p.m_ctx->m_stream);
            break;
        case CUDNN_DATA_HALF:
            catclassifier_fwd_half_i32(
                x_->m_dims[0], (const __half *)x_->deviceMem(),
                (int32_t *)y_->deviceMem(), x_->m_dims[1], p.m_ctx->m_stream);
            break;
        default:
            throw std::runtime_error{"Unsupported tensor datatype"};
        }
    }

    CudaOpArgs listInputs() const override { return {{"x", x_}}; }

    CudaOpArgs listOutputs() const override { return {{"y", y_}}; }
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

    void exec(CudaProgram &p) override
    {
        const int n = x_->m_dims[0];
        const int c = x_->m_dims[1];
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
            throw std::runtime_error{"Unsupported tensor datatype"};
        }
    }

    CudaOpArgs listInputs() const override
    {
        return {{"x", x_}, {"y", y_}, {"dy", dy_}};
    }

    CudaOpArgs listOutputs() const override
    {
        return {{"dx", dx_}, {"loss", loss_}};
    }
};

static void
catclassifier_setup(CudaProgram &p, CudaProgramUnit &pu, const Node &n)
{
    auto x = p.lower_tensor(pu, n.m_inputs["x"]);
    auto y = p.lower_tensor(pu, n.m_outputs["y"]);

    auto op = std::make_shared<CudaCatClassifierFwd>(x, y);

    pu.fwd(op);

    if(p.m_pt == ProgramType::INFERENCE)
        return;

    auto dx = p.lower_grad(pu, n.m_inputs["x"]);
    auto dy = p.lower_grad(pu, n.m_outputs["y"]);

    auto loss = p.lower_tensor(pu, n.m_outputs["loss"]);

    pu.bwd(std::make_shared<CudaCatClassifierBwd>(x, y, dx, dy, loss));
}

REGISTER_CUDA_OP("catclassifier", catclassifier_setup);

//------------------------------------------------------------------------
struct CudaLossFwd : public CudaOperation {
    const std::shared_ptr<CudaTensor> x_, y_;
    const size_t m_elements;

    CudaLossFwd(std::shared_ptr<CudaTensor> x, std::shared_ptr<CudaTensor> y)
      : CudaOperation("lossfwd"), x_(x), y_(y), m_elements(x->m_dims.elements())
    {
    }

    void exec(CudaProgram &p) override
    {
        if(x_->m_type == CUDNN_DATA_HALF && y_->m_type == CUDNN_DATA_FLOAT) {
            convert_half_float(x_->deviceMem(), y_->deviceMem(), m_elements,
                               1.0f, p.m_ctx->m_stream);
        } else if(x_->m_type == CUDNN_DATA_HALF &&
                  y_->m_type == CUDNN_DATA_HALF) {
            // FIXME: Use aliased tensor instead
            cudaMemcpyAsync(y_->deviceMem(), x_->deviceMem(),
                            m_elements * sizeof(int16_t), cudaMemcpyDefault,
                            p.m_ctx->m_stream);
        } else if(x_->m_type == CUDNN_DATA_FLOAT &&
                  y_->m_type == CUDNN_DATA_FLOAT) {
            // FIXME: Use aliased tensor instead
            cudaMemcpyAsync(y_->deviceMem(), x_->deviceMem(),
                            m_elements * sizeof(float), cudaMemcpyDefault,
                            p.m_ctx->m_stream);
        } else {
            throw std::runtime_error{"Unsupported tensor datatype"};
        }
    }

    CudaOpArgs listInputs() const override { return {{"x", x_}}; }

    CudaOpArgs listOutputs() const override { return {{"y", y_}}; }
};

struct CudaLossBwd : public CudaOperation {
    const std::shared_ptr<CudaTensor> x_, dx_, m_target, mmss_;

    const float m_strength;

    CudaLossBwd(std::shared_ptr<CudaTensor> x, std::shared_ptr<CudaTensor> dx,
                std::shared_ptr<CudaTensor> target,
                std::shared_ptr<CudaTensor> mmss, float strength)
      : CudaOperation("lossbwd")
      , x_(x)
      , dx_(dx)
      , m_target(target)
      , mmss_(mmss)
      , m_strength(strength)
    {
    }

    void exec(CudaProgram &p) override
    {
        const int n = x_->m_dims[0];

        int c = 1;
        for(size_t i = 1; i < x_->m_dims.size(); i++) {
            c *= x_->m_dims[i];
        }

        const float scale = m_strength / n;

        if(x_->m_type == CUDNN_DATA_HALF &&
           m_target->m_type == CUDNN_DATA_FLOAT) {
            loss_bwd_half_float(n, (const __half *)x_->deviceMem(),
                                (__half *)dx_->deviceMem(),
                                (const float *)m_target->deviceMem(),
                                (float *)mmss_->deviceMem(), c,
                                scale * p.m_mp_scaling, p.m_ctx->m_stream);

        } else if(x_->m_type == CUDNN_DATA_HALF &&
                  m_target->m_type == CUDNN_DATA_HALF) {
            loss_bwd_half_half(n, (const __half *)x_->deviceMem(),
                               (__half *)dx_->deviceMem(),
                               (const half *)m_target->deviceMem(),
                               (float *)mmss_->deviceMem(), c,
                               scale * p.m_mp_scaling, p.m_ctx->m_stream);
        } else if(x_->m_type == CUDNN_DATA_FLOAT &&
                  m_target->m_type == CUDNN_DATA_FLOAT) {
            loss_bwd_float_float(n, (const float *)x_->deviceMem(),
                                 (float *)dx_->deviceMem(),
                                 (const float *)m_target->deviceMem(),
                                 (float *)mmss_->deviceMem(), c,
                                 scale * p.m_mp_scaling, p.m_ctx->m_stream);
        } else {
            throw std::runtime_error{"Unsupported tensor datatype"};
        }
    }

    CudaOpArgs listInputs() const override
    {
        return {{"x", x_}, {"target", m_target}};
    }

    CudaOpArgs listOutputs() const override
    {
        return {{"dx", dx_}, {"mmss", mmss_}};
    }
};

static void
loss_setup(CudaProgram &p, CudaProgramUnit &pu, const Node &n)
{
    auto x = p.lower_tensor(pu, n.m_inputs["x"]);
    auto y = p.lower_tensor(pu, n.m_outputs["y"]);

    auto op = std::make_shared<CudaLossFwd>(x, y);

    pu.fwd(op);

    if(p.m_pt == ProgramType::INFERENCE)
        return;

    auto dx = p.lower_grad(pu, n.m_inputs["x"]);
    auto target = p.lower_tensor(pu, n.m_inputs["target"]);
    auto mmss = p.lower_tensor(pu, n.m_outputs["mmss"]);
    assert(mmss);

    const float strength = n.m_attributes.get("strength", 1.0f);
    pu.bwd(std::make_shared<CudaLossBwd>(x, dx, target, mmss, strength));
}

REGISTER_CUDA_OP("loss", loss_setup);

//------------------------------------------------------------------------

struct CudnnDropoutFwd : public CudaOperation {
    const std::shared_ptr<CudaContext> ctx_;
    const std::shared_ptr<CudaTensor> x_, y_;
    cudnnDropoutDescriptor_t desc_;
    size_t reserve_size_;
    void *reserve_;
    size_t states_size_;
    void *states_;

    CudnnDropoutFwd(CudaProgram &p, std::shared_ptr<CudaTensor> x,
                    std::shared_ptr<CudaTensor> y, float prob)
      : CudaOperation("dropoutfwd"), ctx_(p.m_ctx), x_(x), y_(y)
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

    void exec(CudaProgram &p) override
    {
        chkCUDNN(cudnnDropoutForward(p.m_ctx->m_cudnn, desc_, x_->desc(),
                                     x_->deviceMem(), y_->desc(),
                                     y_->deviceMem(), reserve_, reserve_size_));
    }

    CudaOpArgs listInputs() const override { return {{"x", x_}}; }

    CudaOpArgs listOutputs() const override { return {{"y", y_}}; }
};

struct CudnnDropoutBwd : public CudaOperation {
    const std::shared_ptr<CudnnDropoutFwd> fwd_;
    const std::shared_ptr<CudaTensor> dx_, dy_;

    CudnnDropoutBwd(std::shared_ptr<CudnnDropoutFwd> fwd,
                    std::shared_ptr<CudaTensor> dx,
                    std::shared_ptr<CudaTensor> dy)
      : CudaOperation("dropoutbwd"), fwd_(fwd), dx_(dx), dy_(dy)
    {
    }

    ~CudnnDropoutBwd() {}

    void exec(CudaProgram &p) override
    {
        chkCUDNN(cudnnDropoutBackward(p.m_ctx->m_cudnn, fwd_->desc_,
                                      dy_->desc(), dy_->deviceMem(),
                                      dx_->desc(), dx_->deviceMem(),
                                      fwd_->reserve_, fwd_->reserve_size_));
    }

    CudaOpArgs listInputs() const override { return {{"dy", dy_}}; }

    CudaOpArgs listOutputs() const override { return {{"dx", dx_}}; }
};

static void
dropout_setup(CudaProgram &p, CudaProgramUnit &pu, const Node &n)
{
    assert(p.m_pt == ProgramType::TRAINING);

    auto x = p.lower_tensor(pu, n.m_inputs["x"]);
    auto y = p.lower_tensor(pu, n.m_outputs["y"]);
    const float prob = n.m_attributes.get("prob", 0.5f);

    auto fwd = std::make_shared<CudnnDropoutFwd>(p, x, y, prob);
    pu.fwd(fwd);

    auto dx = p.lower_grad(pu, n.m_inputs["x"]);
    auto dy = p.lower_grad(pu, n.m_outputs["y"]);

    pu.bwd(std::make_shared<CudnnDropoutBwd>(fwd, dx, dy));
}

REGISTER_CUDA_OP("dropout", dropout_setup);

static std::vector<std::shared_ptr<Node>>
dropout_transform_node(CudaProgram &p, CudaProgramUnit &pu,
                       std::shared_ptr<Node> n)
{
    auto y = n->m_outputs["y"];
    auto ly = p.m_ctx->m_tensors[y];

    if(ly) {
        auto x = n->m_inputs["x"];
        auto lx = std::make_shared<CudaTensor>(ly->m_storage, ly->m_dims,
                                               p.tensorFormat(*ly),
                                               ly->namePostfix("dropout"));
        p.m_ctx->m_tensors[x] = ly;

    } else {
        auto x = p.lower_tensor(pu, n->m_inputs["x"]);
        ly = std::make_shared<CudaTensor>(x->m_storage, x->m_dims,
                                          p.tensorFormat(*x),
                                          x->namePostfix("dropout"));
        p.m_ctx->m_tensors[y] = ly;
    }
    return {};
}

static Nodes
dropout_transform(CudaProgram &p, CudaProgramUnit &pu, const Nodes &nodes)
{
    Nodes r;

    for(size_t i = 0; i < nodes.size(); i++) {
        auto &n = nodes[i];
        if(n->m_type == "dropout") {
            dropout_transform_node(p, pu, n);
        } else {
            r.push_back(n);
        }
    }
    return r;
}

REGISTER_CUDA_TRANSFORM(500, ProgramType::INFERENCE, dropout_transform);

//------------------------------------------------------------------------

struct CudnnBatchNormInference : public CudaOperation {
    const std::shared_ptr<CudaTensor> x_, s_, b_, m_, v_, y_;
    const float epsilon_;

    CudnnBatchNormInference(std::shared_ptr<CudaTensor> x,
                            std::shared_ptr<CudaTensor> s,
                            std::shared_ptr<CudaTensor> b,
                            std::shared_ptr<CudaTensor> m,
                            std::shared_ptr<CudaTensor> v,
                            std::shared_ptr<CudaTensor> y, float epsilon)
      : CudaOperation("bninf")
      , x_(x)
      , s_(s)
      , b_(b)
      , m_(m)
      , v_(v)
      , y_(y)
      , epsilon_(epsilon)
    {
    }

    void exec(CudaProgram &p) override
    {
        float alpha = 1.0f;
        float beta = 0.0f;

        chkCUDNN(cudnnBatchNormalizationForwardInference(
            p.m_ctx->m_cudnn, CUDNN_BATCHNORM_SPATIAL, &alpha, &beta,
            x_->desc(), x_->deviceMem(), y_->desc(), y_->deviceMem(),
            s_->desc(), s_->deviceMem(), b_->deviceMem(), m_->deviceMem(),
            v_->deviceMem(), epsilon_));
    }

    CudaOpArgs listInputs() const override
    {
        return {{"x", x_}, {"s", s_}, {"b", b_}, {"m", m_}, {"v", v_}};
    }

    CudaOpArgs listOutputs() const override { return {{"y", y_}}; }
};

struct CudnnBatchNormTrain : public CudaOperation {
    const std::shared_ptr<CudaTensor> x_, s_, b_, m_, v_, y_, sm_, sv_;
    const float epsilon_;
    float expavgf_{NAN};
    float m_m_beta{1.0f};
    float m_v_beta{1.0f};

    CudnnBatchNormTrain(std::shared_ptr<CudaTensor> x,
                        std::shared_ptr<CudaTensor> s,
                        std::shared_ptr<CudaTensor> b,
                        std::shared_ptr<CudaTensor> m,
                        std::shared_ptr<CudaTensor> v,
                        std::shared_ptr<CudaTensor> y,
                        std::shared_ptr<CudaTensor> sm,
                        std::shared_ptr<CudaTensor> sv, float epsilon)
      : CudaOperation("bntrain")
      , x_(x)
      , s_(s)
      , b_(b)
      , m_(m)
      , v_(v)
      , y_(y)
      , sm_(sm)
      , sv_(sv)
      , epsilon_(epsilon)
    {
    }

    void exec(CudaProgram &p) override
    {
        float alpha = 1.0f;
        float beta = 0.0f;

        if(!isfinite(expavgf_)) {
            expavgf_ = m_->m_auto_initialized ? 1.0f : p.m_pc.bn_expavg;
        }

        assert(m_m_beta == 1.0f);
        assert(m_v_beta == 1.0f);

        chkCUDNN(cudnnBatchNormalizationForwardTraining(
            p.m_ctx->m_cudnn, CUDNN_BATCHNORM_SPATIAL, &alpha, &beta,
            x_->desc(), x_->deviceMem(), y_->desc(), y_->deviceMem(),
            s_->desc(), s_->deviceMem(), b_->deviceMem(), expavgf_,
            m_->deviceMem(), v_->deviceMem(), epsilon_, sm_->deviceMem(),
            sv_->deviceMem()));

        expavgf_ = p.m_pc.bn_expavg;
    }

    CudaOpArgs listInputs() const override
    {
        return {{"x", x_}, {"s", s_}, {"b", b_}};
    }

    CudaOpArgs listOutputs() const override
    {
        return {{"y", y_}, {"m", m_}, {"v", v_}, {"sm", sm_}, {"sv", sv_}};
    }

    float *getBeta(const std::shared_ptr<CudaTensor> &t) override
    {
        if(t == m_)
            return &m_m_beta;
        if(t == v_)
            return &m_v_beta;
        return nullptr;
    }
};

struct CudnnBatchNormBwd : public CudaOperation {
    const std::shared_ptr<CudaTensor> x_, dy_, dx_, s_, ds_, db_, sm_, sv_;
    const float epsilon_;
    float m_dx_beta{0};
    float m_ds_beta{0};
    float m_db_beta{0};
    CudnnBatchNormBwd(CudnnBatchNormTrain &fwd, std::shared_ptr<CudaTensor> dy,
                      std::shared_ptr<CudaTensor> dx,
                      std::shared_ptr<CudaTensor> ds,
                      std::shared_ptr<CudaTensor> db)
      : CudaOperation("bnbwd")
      , x_(fwd.x_)
      , dy_(dy)
      , dx_(dx)
      , s_(fwd.s_)
      , ds_(ds)
      , db_(db)
      , sm_(fwd.sm_)
      , sv_(fwd.sv_)
      , epsilon_(fwd.epsilon_)
    {
    }

    void exec(CudaProgram &p) override
    {
        float alpha = 1.0f;

        assert(m_ds_beta == m_db_beta);

        chkCUDNN(cudnnBatchNormalizationBackward(
            p.m_ctx->m_cudnn, CUDNN_BATCHNORM_SPATIAL, &alpha, &m_dx_beta,
            &alpha, &m_ds_beta, x_->desc(), x_->deviceMem(), dy_->desc(),
            dy_->deviceMem(), dx_->desc(), dx_->deviceMem(), s_->desc(),
            s_->deviceMem(), ds_->deviceMem(), db_->deviceMem(), epsilon_,
            sm_->deviceMem(), sv_->deviceMem()));
    }

    CudaOpArgs listInputs() const override
    {
        return {{"x", x_}, {"dy", dy_}, {"s", s_}, {"sm", sm_}, {"sv", sv_}};
    }

    CudaOpArgs listOutputs() const override
    {
        return {{"dx", dx_}, {"ds", ds_}, {"db", db_}};
    }

    float *getBeta(const std::shared_ptr<CudaTensor> &t) override
    {
        if(t == dx_)
            return &m_dx_beta;
        if(t == ds_)
            return &m_ds_beta;
        if(t == db_)
            return &m_db_beta;
        return nullptr;
    }
};

static void
batchnorm_setup(CudaProgram &p, CudaProgramUnit &pu, const Node &n)
{
    auto x = p.lower_tensor(pu, n.m_inputs["x"]);
    auto y = p.lower_tensor(pu, n.m_outputs["y"]);
    auto s = p.lower_tensor(pu, n.m_inputs["s"], 2);
    auto b = p.lower_tensor(pu, n.m_inputs["b"], 2);

    auto m = p.lower_tensor(pu, n.m_inputs["m"], 2);
    auto v = p.lower_tensor(pu, n.m_inputs["v"], 2);
    const float epsilon = n.m_attributes.get("epsilon", 1e-5f);

    if(p.m_pt == ProgramType::INFERENCE) {
        pu.fwd(std::make_shared<CudnnBatchNormInference>(x, s, b, m, v, y,
                                                         epsilon));
        return;
    }

    auto sm = std::make_shared<CudaTensor>(*m, m->namePostfix("smean"));
    auto sv = std::make_shared<CudaTensor>(*v, v->namePostfix("svar"));

    auto f = std::make_shared<CudnnBatchNormTrain>(x, s, b, m, v, y, sm, sv,
                                                   epsilon);
    pu.fwd(f);

    auto dx = p.lower_grad(pu, n.m_inputs["x"]);
    auto dy = p.lower_grad(pu, n.m_outputs["y"]);
    auto ds = p.lower_grad(pu, n.m_inputs["s"], 2);
    auto db = p.lower_grad(pu, n.m_inputs["b"], 2);

    pu.bwd(std::make_shared<CudnnBatchNormBwd>(*f, dy, dx, ds, db));

    p.upd(n.m_inputs["s"], s, ds);
    p.upd(n.m_inputs["b"], b, db);
}

REGISTER_CUDA_OP("batchnorm", batchnorm_setup);

//------------------------------------------------------------------------

struct CudnnOpTensor : public CudaOperation {
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
            throw std::runtime_error(
                fmt("Unsupported op %d in %s", (int)op, __PRETTY_FUNCTION__));
        }
    }

    CudnnOpTensor(std::shared_ptr<CudaTensor> a, std::shared_ptr<CudaTensor> b,
                  std::shared_ptr<CudaTensor> c, cudnnOpTensorOp_t op)
      : CudaOperation(opname(op)), a_(a), b_(b), c_(c)
    {
        cudnnCreateOpTensorDescriptor(&desc_);
        cudnnSetOpTensorDescriptor(desc_, op, CUDNN_DATA_FLOAT,
                                   CUDNN_PROPAGATE_NAN);
    }

    ~CudnnOpTensor() { cudnnDestroyOpTensorDescriptor(desc_); }

    void exec(CudaProgram &p) override
    {
        float alpha = 1.0f;
        float beta = 0.0f;

        chkCUDNN(cudnnOpTensor(p.m_ctx->m_cudnn, desc_, &alpha, a_->desc(),
                               a_->deviceMem(), &alpha, b_->desc(),
                               b_->deviceMem(), &beta, c_->desc(),
                               c_->deviceMem()));
    }

    CudaOpArgs listInputs() const override { return {{"a", a_}, {"b", b_}}; }

    CudaOpArgs listOutputs() const override { return {{"c", c_}}; }
};

static void
sum_setup(CudaProgram &p, CudaProgramUnit &pu, const Node &n)
{
    auto x0 = p.lower_tensor(pu, n.m_inputs["x0"]);
    auto x1 = p.lower_tensor(pu, n.m_inputs["x1"]);

    auto y = p.lower_tensor(pu, n.m_outputs["y"]);

    auto fwd = std::make_shared<CudnnOpTensor>(x0, x1, y, CUDNN_OP_TENSOR_ADD);

    pu.fwd(fwd);

    if(p.m_pt == ProgramType::INFERENCE)
        return;

    auto dy = p.lower_grad(pu, n.m_outputs["y"]);

    auto dx0 = p.lower_grad(pu, n.m_inputs["x0"]);
    pu.bwd(std::make_shared<CudnnTransform>(dy, dx0));

    auto dx1 = p.lower_grad(pu, n.m_inputs["x1"]);
    pu.bwd(std::make_shared<CudnnTransform>(dy, dx1));
}

REGISTER_CUDA_OP("sum", sum_setup);

static void
add_setup(CudaProgram &p, CudaProgramUnit &pu, const Node &n)
{
    auto x = p.lower_tensor(pu, n.m_inputs["x"]);
    auto y = p.lower_tensor(pu, n.m_outputs["y"]);
    auto b = p.lower_tensor(pu, n.m_inputs["b"]);

    auto fwd = std::make_shared<CudnnOpTensor>(x, b, y, CUDNN_OP_TENSOR_ADD);

    if(p.m_pt == ProgramType::INFERENCE) {
        pu.fwd(fwd);
        return;
    }
    throw std::runtime_error{"Add not supported for backprop"};
}

REGISTER_CUDA_OP("add", add_setup);

//------------------------------------------------------------------------

struct CudnnSoftmaxFwd : public CudaOperation {
    const std::shared_ptr<CudaTensor> x_, y_;

    CudnnSoftmaxFwd(std::shared_ptr<CudaTensor> x,
                    std::shared_ptr<CudaTensor> y)
      : CudaOperation("softmaxfwd"), x_(x), y_(y)
    {
    }

    void exec(CudaProgram &p) override
    {
        float alpha = 1.0f, beta = 0.0f;

        chkCUDNN(cudnnSoftmaxForward(p.m_ctx->m_cudnn, CUDNN_SOFTMAX_ACCURATE,
                                     CUDNN_SOFTMAX_MODE_CHANNEL, &alpha,
                                     x_->desc(), x_->deviceMem(), &beta,
                                     y_->desc(), y_->deviceMem()));
    }

    CudaOpArgs listInputs() const override { return {{"x", x_}}; }

    CudaOpArgs listOutputs() const override { return {{"y", y_}}; }
};

static void
softmax_setup(CudaProgram &p, CudaProgramUnit &pu, const Node &n)
{
    if(p.m_pt != ProgramType::INFERENCE) {
        throw std::runtime_error{"not supported for training"};
    }

    auto x = p.lower_tensor(pu, n.m_inputs["x"]);
    auto y = p.lower_tensor(pu, n.m_outputs["y"]);
    pu.fwd(std::make_shared<CudnnSoftmaxFwd>(x, y));
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

struct CudnnBatchNormActTrain : public CudaOperation {
    const std::shared_ptr<CudaContext> ctx_;
    const std::shared_ptr<CudaTensor> x_, z_, s_, b_, m_, v_, y_, sm_, sv_;
    const float epsilon_;
    float expavgf_{NAN};
    float m_m_beta{1.0f};
    float m_v_beta{1.0f};
    const cudnnBatchNormOps_t ops_;
    const cudnnBatchNormMode_t mode_;
    cudnnActivationDescriptor_t desc_;

    std::shared_ptr<CudaTensor> m_reserve;

    CudnnBatchNormActTrain(
        CudaProgram &p, std::shared_ptr<CudaTensor> x,
        std::shared_ptr<CudaTensor> z, std::shared_ptr<CudaTensor> s,
        std::shared_ptr<CudaTensor> b, std::shared_ptr<CudaTensor> m,
        std::shared_ptr<CudaTensor> v, std::shared_ptr<CudaTensor> y,
        std::shared_ptr<CudaTensor> sm, std::shared_ptr<CudaTensor> sv,
        float epsilon, cudnnBatchNormOps_t ops,
        cudnnActivationMode_t activation_mode, float actalpha)
      : CudaOperation(std::string(bnopsstr(ops)) + "_fwd.persistent")
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
      , ops_(ops)
      , mode_(CUDNN_BATCHNORM_SPATIAL_PERSISTENT)
    {
        chkCUDNN(cudnnCreateActivationDescriptor(&desc_));
        chkCUDNN(cudnnSetActivationDescriptor(desc_, activation_mode,
                                              CUDNN_PROPAGATE_NAN, actalpha));

        size_t reserve_size;

        chkCUDNN(cudnnGetBatchNormalizationTrainingExReserveSpaceSize(
            p.m_ctx->m_cudnn, mode_, ops_, desc_, x_->desc(), &reserve_size));

        if(reserve_size > 0) {
            assert(reserve_size < INT32_MAX);
            m_reserve = std::make_shared<CudaTensor>(
                Tensor::DataType::U8, Dims{{(int)reserve_size, 1, 1, 1}},
                CUDNN_TENSOR_NCHW, p.m_ctx, "bnr");
        }
    }

    ~CudnnBatchNormActTrain()
    {
        chkCUDNN(cudnnDestroyActivationDescriptor(desc_));
    }

    void exec(CudaProgram &p) override
    {
        float alpha = 1.0f;
        float beta = 0.0f;

        void *reserve = m_reserve ? m_reserve->m_storage->m_data : NULL;
        size_t reserve_size = m_reserve ? (size_t)m_reserve->m_dims[0] : 0;

        if(!isfinite(expavgf_)) {
            expavgf_ = m_->m_auto_initialized ? 1.0f : p.m_pc.bn_expavg;
        }

        assert(m_m_beta == 1.0f);
        assert(m_v_beta == 1.0f);

        chkCUDNN(cudnnBatchNormalizationForwardTrainingEx(
            ctx_->m_cudnn, mode_, ops_, &alpha, &beta, x_->desc(),
            x_->deviceMem(), z_ ? z_->desc() : NULL,
            z_ ? z_->deviceMem() : NULL, y_->desc(), y_->deviceMem(),
            s_->desc(), s_->deviceMem(), b_->deviceMem(), expavgf_,
            m_->deviceMem(), v_->deviceMem(), epsilon_, sm_->deviceMem(),
            sv_->deviceMem(), desc_, ctx_->m_workspace.ptr(),
            ctx_->m_workspace.size(), reserve, reserve_size));

        expavgf_ = p.m_pc.bn_expavg;

        if(!p.m_pc.anomaly_detect)
            return;
    }

    CudaOpArgs listInputs() const override
    {
        CudaOpArgs r;
        r.push_back({"x", x_});
        r.push_back({"s", s_});
        r.push_back({"b", b_});

        if(z_)
            r.push_back({"z", z_});
        return r;
    }

    CudaOpArgs listOutputs() const override
    {
        CudaOpArgs r;
        r.push_back({"y", y_});
        r.push_back({"m", m_});
        r.push_back({"v", v_});
        r.push_back({"sm", sm_});
        r.push_back({"sv", sv_});

        if(m_reserve)
            r.push_back({"res", m_reserve});
        return r;
    }

    float *getBeta(const std::shared_ptr<CudaTensor> &t) override
    {
        if(t == m_)
            return &m_m_beta;
        if(t == v_)
            return &m_v_beta;
        return nullptr;
    }
};

struct CudnnBatchNormActBwd : public CudaOperation {
    const std::shared_ptr<CudnnBatchNormActTrain> fwd_;
    std::shared_ptr<CudaTensor> dy_, dx_, dz_, ds_, db_;
    float m_dx_beta{0};
    float m_ds_beta{0};
    float m_db_beta{0};

    cudnnBatchNormOps_t ops_;

    CudnnBatchNormActBwd(
        CudaProgram &p, std::shared_ptr<CudnnBatchNormActTrain> fwd,
        std::shared_ptr<CudaTensor> dy, std::shared_ptr<CudaTensor> dx,
        std::shared_ptr<CudaTensor> dz, std::shared_ptr<CudaTensor> ds,
        std::shared_ptr<CudaTensor> db, cudnnBatchNormOps_t ops)
      : CudaOperation(std::string(bnopsstr(ops)) + "_bwd.persistent")
      , fwd_(fwd)
      , dy_(dy)
      , dx_(dx)
      , dz_(dz)
      , ds_(ds)
      , db_(db)
      , ops_(ops)
    {
        size_t workspace;
        chkCUDNN(cudnnGetBatchNormalizationBackwardExWorkspaceSize(
            p.m_ctx->m_cudnn, fwd->mode_, fwd->ops_, fwd->x_->desc(),
            fwd->y_->desc(), dy_->desc(), dz_ ? dz_->desc() : NULL, dx_->desc(),
            ds_->desc(), fwd->desc_, &workspace));
        p.m_ctx->m_workspace.request(workspace);
    }

    void exec(CudaProgram &p) override
    {
        float alpha = 1.0f;

        void *reserve =
            fwd_->m_reserve ? fwd_->m_reserve->m_storage->m_data : NULL;
        size_t reserve_size =
            fwd_->m_reserve ? (size_t)fwd_->m_reserve->m_dims[0] : 0;

        assert(m_ds_beta == m_db_beta);

        chkCUDNN(cudnnBatchNormalizationBackwardEx(
            p.m_ctx->m_cudnn, fwd_->mode_, ops_, &alpha, &m_dx_beta, &alpha,
            &m_ds_beta, fwd_->x_->desc(), fwd_->x_->deviceMem(),
            fwd_->y_->desc(), fwd_->y_->deviceMem(), dy_->desc(),
            dy_->deviceMem(), dz_ ? dz_->desc() : NULL,
            dz_ ? dz_->deviceMem() : NULL, dx_->desc(), dx_->deviceMem(),
            fwd_->s_->desc(), fwd_->s_->deviceMem(), fwd_->b_->deviceMem(),
            ds_->deviceMem(), db_->deviceMem(), fwd_->epsilon_,
            fwd_->sm_->deviceMem(), fwd_->sv_->deviceMem(), fwd_->desc_,
            p.m_ctx->m_workspace.ptr(), p.m_ctx->m_workspace.size(), reserve,
            reserve_size));

        if(!p.m_pc.anomaly_detect)
            return;
    }

    CudaOpArgs listInputs() const override
    {
        CudaOpArgs r;
        r.push_back({"x", fwd_->x_});
        r.push_back({"y", fwd_->y_});
        r.push_back({"dy", dy_});
        r.push_back({"s", fwd_->s_});
        r.push_back({"b", fwd_->b_});
        r.push_back({"sm", fwd_->sm_});
        r.push_back({"sv", fwd_->sv_});

        if(fwd_->m_reserve)
            r.push_back({"res", fwd_->m_reserve});
        return r;
    }

    CudaOpArgs listOutputs() const override
    {
        CudaOpArgs r;
        r.push_back({"dx", dx_});
        r.push_back({"ds", ds_});
        r.push_back({"db", db_});
        if(dz_)
            r.push_back({"dz", dz_});
        return r;
    }

    virtual bool killOutput(std::shared_ptr<CudaTensorStorage> s) override
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

    float *getBeta(const std::shared_ptr<CudaTensor> &t) override
    {
        if(t == dx_)
            return &m_dx_beta;
        if(t == ds_)
            return &m_ds_beta;
        if(t == db_)
            return &m_db_beta;
        return nullptr;
    }
};

static void
batchnorm_persistent_setup(CudaProgram &p, CudaProgramUnit &pu, const Node &n)
{
    if(p.m_pt == ProgramType::INFERENCE) {
        throw std::runtime_error{"not supported for inferenece"};
    }

    auto x0 = p.lower_tensor(pu, n.m_inputs["x0"]);
    auto x1 =
        n.m_inputs.has("x1") ? p.lower_tensor(pu, n.m_inputs["x1"]) : nullptr;
    auto y = p.lower_tensor(pu, n.m_outputs["y"]);
    auto s = p.lower_tensor(pu, n.m_inputs["s"], 2);
    auto b = p.lower_tensor(pu, n.m_inputs["b"], 2);

    auto m = p.lower_tensor(pu, n.m_inputs["m"], 2);
    auto v = p.lower_tensor(pu, n.m_inputs["v"], 2);
    const float epsilon = n.m_attributes.get("epsilon", 1e-5f);

    auto sm = std::make_shared<CudaTensor>(*m, m->namePostfix("smean"));
    auto sv = std::make_shared<CudaTensor>(*v, v->namePostfix("svar"));

    auto ops = CUDNN_BATCHNORM_OPS_BN;
    if(n.m_attributes.get("relu", false)) {
        ops = CUDNN_BATCHNORM_OPS_BN_ACTIVATION;
        if(x1)
            ops = CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION;
    } else {
        assert(x1 == nullptr);
    }

    auto activation_mode = CUDNN_ACTIVATION_RELU;
    float activation_alpha = 0.0f;

    auto f = std::make_shared<CudnnBatchNormActTrain>(
        p, x0, x1, s, b, m, v, y, sm, sv, epsilon, ops, activation_mode,
        activation_alpha);
    pu.fwd(f);

    auto dx0 = p.lower_grad(pu, n.m_inputs["x0"]);
    auto dx1 = x1 ? p.lower_grad(pu, n.m_inputs["x1"]) : nullptr;
    auto dy = p.lower_grad(pu, n.m_outputs["y"]);
    auto ds = p.lower_grad(pu, n.m_inputs["s"], 2);
    auto db = p.lower_grad(pu, n.m_inputs["b"], 2);

    pu.bwd(std::make_shared<CudnnBatchNormActBwd>(p, f, dy, dx0, dx1, ds, db,
                                                  ops));

    p.upd(n.m_inputs["s"], s, ds);
    p.upd(n.m_inputs["b"], b, db);
}

REGISTER_CUDA_OP("batchnorm.persistent", batchnorm_persistent_setup);

static std::shared_ptr<Node>
batchnorm_persistent_transform_node(CudaProgram &p, std::shared_ptr<Node> bn,
                                    std::shared_ptr<Node> sum,
                                    std::shared_ptr<Node> relu)
{
    auto x = bn->m_inputs["x"];

    if(x->m_data_type != Tensor::DataType::HALF)
        return nullptr;

    if(x->m_dims[1] % 4)
        return nullptr;

    auto lx = p.m_ctx->m_tensors.find(x);
    if(lx != p.m_ctx->m_tensors.end()) {
        if(!lx->second->cpacked())
            return nullptr;
    }

    auto y = relu ? relu->m_outputs["y"] : bn->m_outputs["y"];

    auto ly = p.m_ctx->m_tensors.find(y);
    if(ly != p.m_ctx->m_tensors.end()) {
        if(!ly->second->cpacked())
            return nullptr;
    }

    auto nn = std::make_shared<Node>("batchnorm.persistent");

    nn->m_inputs["x0"] = bn->m_inputs["x"];
    auto bn_y = bn->m_outputs["y"];

    if(sum) {
        auto sum_x0 = sum->m_inputs["x0"];
        auto sum_x1 = sum->m_inputs["x1"];

        if(sum_x0 == bn_y) {
            nn->m_inputs["x1"] = sum->m_inputs["x1"];
        } else {
            nn->m_inputs["x1"] = sum->m_inputs["x0"];
        }
    }
    nn->m_inputs["s"] = bn->m_inputs["s"];
    nn->m_inputs["b"] = bn->m_inputs["b"];
    nn->m_inputs["m"] = bn->m_inputs["m"];
    nn->m_inputs["v"] = bn->m_inputs["v"];

    if(bn->m_attributes.find("epsilon") != bn->m_attributes.end())
        nn->m_attributes["epsilon"] = bn->m_attributes["epsilon"];

    if(relu)
        nn->m_attributes["relu"] = true;

    nn->m_outputs["y"] = y;
    return nn;
}

static Nodes
batchnorm_relu_transform(CudaProgram &p, CudaProgramUnit &pu,
                         const Nodes &nodes)
{
    if(p.m_pc.disable_op_fusing)
        return nodes;

    Nodes r;
    const ssize_t num_nodes = nodes.size();

    for(ssize_t i = 0; i < num_nodes; i++) {
        std::shared_ptr<Node> n = nodes[i];

        if(i < num_nodes - 1 && nodes[i + 0]->m_type == "batchnorm" &&
           nodes[i + 1]->m_type == "relu") {
            auto n2 = batchnorm_persistent_transform_node(p, nodes[i], nullptr,
                                                          nodes[i + 1]);
            if(n2) {
                i++;
                n = n2;
            }
        } else if(i < num_nodes - 2 && nodes[i + 0]->m_type == "batchnorm" &&
                  nodes[i + 1]->m_type == "sum" &&
                  nodes[i + 2]->m_type == "relu") {
            auto n2 = batchnorm_persistent_transform_node(
                p, nodes[i], nodes[i + 1], nodes[i + 2]);
            if(n2) {
                i += 2;
                n = n2;
            }
        } else if(nodes[i]->m_type == "batchnorm") {
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

REGISTER_CUDA_TRANSFORM(500, ProgramType::TRAINING, batchnorm_relu_transform);

//------------------------------------------------------------------------

struct CudnnSpatialTransformFwd : public CudaOperation {
    const std::shared_ptr<CudaTensor> x_, theta_, y_, grid_;
    cudnnSpatialTransformerDescriptor_t desc_;

    CudnnSpatialTransformFwd(std::shared_ptr<CudaContext> ctx,
                             std::shared_ptr<CudaTensor> x,
                             std::shared_ptr<CudaTensor> theta,
                             std::shared_ptr<CudaTensor> y)
      : CudaOperation("spatialtransform")
      , x_(x)
      , theta_(theta)
      , y_(y)
      , grid_(std::make_shared<CudaTensor>(
            Tensor::DataType::FLOAT,
            Dims{y_->m_dims[0], 2, y_->m_dims[2], y_->m_dims[3]},
            CUDNN_TENSOR_NHWC, ctx))
    {
        int dims[4] = {
            (int)y_->m_dims[0],  // n
            1,                   // c
            (int)y_->m_dims[2],  // h
            (int)y_->m_dims[3]   // w
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

    void exec(CudaProgram &p) override
    {
        float alpha = 1.0f, beta = 0.0f;

        chkCUDNN(cudnnSpatialTfGridGeneratorForward(
            p.m_ctx->m_cudnn, desc_, theta_->deviceMem(), grid_->deviceMem()));
        chkCUDNN(cudnnSpatialTfSamplerForward(
            p.m_ctx->m_cudnn, desc_, &alpha, x_->desc(), x_->deviceMem(),
            grid_->deviceMem(), &beta, y_->desc(), y_->deviceMem()));
    }

    CudaOpArgs listInputs() const override
    {
        return {{"x", x_}, {"theta", theta_}};
    }

    CudaOpArgs listOutputs() const override { return {{"y", y_}}; }
};

static void
spatialtransform_setup(CudaProgram &p, CudaProgramUnit &pu, const Node &n)
{
    auto x = p.lower_tensor(pu, n.m_inputs["x"]);
    auto y = p.lower_tensor(pu, n.m_outputs["y"]);

    const bool disable = p.m_pt == ProgramType::INFERENCE &&
                         !n.m_attributes.get("inference", false);
    const bool bypass = disable && x->m_dims == y->m_dims;

    if(bypass) {
        pu.fwd(std::make_shared<CudnnTransform>(x, y));
        return;
    }

    auto th = n.m_inputs.has("theta") ? n.m_inputs["theta"] : nullptr;

    std::shared_ptr<CudaTensor> theta;

    if(disable || !th) {
        th = makeCPUTensor(Tensor::DataType::FLOAT,
                           Dims({pu.m_batch_size, 2, 3}), "theta.identity");
        auto ta = th->access();
        for(int i = 0; i < pu.m_batch_size; i++) {
            ta->set({i, 0, 0}, 1);
            ta->set({i, 1, 1}, 1);
        }
    }
    theta = p.lower_tensor(pu, th);

    auto op = std::make_shared<CudnnSpatialTransformFwd>(p.m_ctx, x, theta, y);
    pu.fwd(op);
}

REGISTER_CUDA_OP("spatialtransform", spatialtransform_setup);

static Nodes
spatialtransform_transform(CudaProgram &p, CudaProgramUnit &pu,
                           const Nodes &nodes)
{
    if(p.m_pc.disable_op_fusing)
        return nodes;

    Nodes r;

    for(size_t i = 0; i < nodes.size(); i++) {
        auto &n = nodes[i];
        if(n->m_type == "spatialtransform") {
            auto x = n->m_inputs["x"];
            if(x && x->m_data_type != Tensor::DataType::FLOAT) {
                Tensors emptyset;
                auto n0 = Node::make(
                    "convert", {{"x", x}},
                    {{"datatype", (int)Tensor::DataType::FLOAT}}, emptyset)[0];
                auto n1 = Node::make(
                    "spatialtransform",
                    {{"x", n0->y()}, {"theta", n->m_inputs["theta"]}},
                    n->m_attributes, emptyset)[0];
                auto n2 = Node::make("convert", {{"x", n1->y()}},
                                     {{"datatype", (int)x->m_data_type}},
                                     emptyset)[0];
                n2->m_outputs["y"] = n->m_outputs["y"];
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

REGISTER_CUDA_TRANSFORM(120, ProgramType::INFERENCE | ProgramType::TRAINING,
                        spatialtransform_transform);

//------------------------------------------------------------------------

static std::vector<std::shared_ptr<Node>>
reshape_transform_node(CudaProgram &p, CudaProgramUnit &pu,
                       std::shared_ptr<Node> n)
{
    auto xh = n->m_inputs["x"];
    auto yh = n->m_outputs["y"];

    auto fmt = p.tensorFormat(xh->m_data_type);
    auto x = p.lower_tensor(pu, xh, fmt);

    auto y = std::make_shared<CudaTensor>(x->m_storage,
                                          yh->m_dims.batch(pu.m_batch_size),
                                          fmt, x->namePostfix("reshape"));

    p.m_ctx->m_tensors[yh] = y;

    auto dx = p.lower_grad(pu, xh, fmt);
    auto dy = std::make_shared<CudaTensor>(dx->m_storage,
                                           y->m_dims.batch(pu.m_batch_size),
                                           fmt, x->namePostfix("reshape"));
    p.m_ctx->m_tensors[yh->grad()] = dy;
    return {};
}

static Nodes
reshape_transform(CudaProgram &p, CudaProgramUnit &pu, const Nodes &nodes)
{
    Nodes r;

    for(size_t i = 0; i < nodes.size(); i++) {
        auto &n = nodes[i];
        if(n->m_type == "reshape") {
            reshape_transform_node(p, pu, n);
        } else {
            r.push_back(n);
        }
    }
    return r;
}

REGISTER_CUDA_TRANSFORM(110, ProgramType::INFERENCE | ProgramType::TRAINING,
                        reshape_transform);

//------------------------------------------------------------------------

static void
window_transform_node(CudaProgram &p, CudaProgramUnit &pu, const Node &n)
{
    auto x = p.lower_tensor(pu, n.m_inputs["x"], CUDNN_TENSOR_NCHW);
    auto dx = p.lower_grad(pu, n.m_inputs["x"], CUDNN_TENSOR_NCHW);
    auto y = n.m_outputs["y"];

    Dims offset(n.m_attributes.get("offset", std::vector<int>{}));

    auto yl =
        std::make_shared<CudaTensor>(x, y->m_dims.batch(pu.m_batch_size),
                                     offset.i64(), y->namePostfix("alias"));

    p.m_ctx->m_tensors[y] = yl;

    auto dxl =
        std::make_shared<CudaTensor>(dx, y->m_dims.batch(pu.m_batch_size),
                                     offset.i64(), y->namePostfix("alias"));
    p.m_ctx->m_tensors[dx] = dxl;
}

static Nodes
window_transform(CudaProgram &p, CudaProgramUnit &pu, const Nodes &nodes)
{
    Nodes r;

    for(size_t i = 0; i < nodes.size(); i++) {
        auto &n = nodes[i];
        if(n->m_type == "window") {
            window_transform_node(p, pu, *n);
        } else {
            r.push_back(n);
        }
    }
    return r;
}

REGISTER_CUDA_TRANSFORM(100, ProgramType::INFERENCE | ProgramType::TRAINING,
                        window_transform);

//------------------------------------------------------------------------

static void
concat_transform_node(CudaProgram &p, CudaProgramUnit &pu, const Node &n)
{
    const int axis = n.m_attributes.get("axis", 1);

    auto y = p.lower_tensor(pu, n.m_outputs["y"]);
    auto dy = p.lower_grad(pu, n.m_outputs["y"]);
    auto element_offset = std::vector<int64_t>(y->m_dims.size(), 0);

    for(const auto &xh : n.m_inputs.getv("x")) {
        auto x = std::make_shared<CudaTensor>(
            y, xh->m_dims.batch(pu.m_batch_size), element_offset,
            xh->namePostfix("alias"));
        x->copyFromLocked(*xh);
        p.m_ctx->m_tensors[xh] = x;

        auto dx = std::make_shared<CudaTensor>(
            dy, xh->m_dims.batch(pu.m_batch_size), element_offset,
            xh->namePostfix("alias"));

        p.m_ctx->m_tensors[xh->grad()] = dx;

        element_offset[axis] += xh->m_dims[axis];
    }
}

static Nodes
concat_transform(CudaProgram &p, CudaProgramUnit &pu, const Nodes &nodes)
{
    Nodes r;

    for(ssize_t i = nodes.size() - 1; i >= 0; i--) {
        auto &n = nodes[i];
        if(n->m_type == "concat") {
            concat_transform_node(p, pu, *n);
        } else {
            r.insert(r.begin(), n);
        }
    }
    return r;
}

REGISTER_CUDA_TRANSFORM(100, ProgramType::INFERENCE | ProgramType::TRAINING,
                        concat_transform);

//------------------------------------------------------------------------

struct CudaStats : public CudaOperation {
    const std::shared_ptr<CudaTensor> x_, y_;
    const size_t m_elements;
    CudaStats(std::shared_ptr<CudaTensor> x, std::shared_ptr<CudaTensor> y)
      : CudaOperation("stats"), x_(x), y_(y), m_elements(x->m_dims.elements())
    {
    }

    void exec(CudaProgram &p) override
    {
        switch(x_->m_data_type) {
        case Tensor::DataType::FLOAT:
            tensor_stats_float(m_elements, (const float *)x_->deviceMem(),
                               (float *)y_->deviceMem(), p.m_ctx->m_stream);
            break;
        case Tensor::DataType::HALF:
            tensor_stats_half(m_elements, (const __half *)x_->deviceMem(),
                              (float *)y_->deviceMem(), p.m_ctx->m_stream);
            break;
        default:
            throw std::runtime_error{"Unsupported datatype"};
        }
    }

    CudaOpArgs listInputs() const override { return {{"x", x_}}; }

    CudaOpArgs listOutputs() const override { return {{"y", y_}}; }
};

static void
stats_setup(CudaProgram &p, CudaProgramUnit &pu, const Node &n)
{
    auto it = p.m_ctx->m_tensors.find(n.m_inputs["x"]);
    if(it == p.m_ctx->m_tensors.end())
        throw std::runtime_error{"x-tensor not found"};

    auto x = it->second;

    auto y = p.lower_tensor(pu, n.m_outputs["y"]);
    pu.tail(std::make_shared<CudaStats>(x, y));
}

REGISTER_CUDA_OP("stats", stats_setup);

}  // namespace saga
