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

struct Optimizer : public CudaOperation {
    CudaOpArgs m_inputs;
    CudaOpArgs m_outputs;
    const std::shared_ptr<CudaTensorStorageMemory> m_mem;
    int m_iter{0};
    const std::shared_ptr<CudaContext> m_ctx;
    const size_t m_elements;

    Optimizer(CudaProgram &p, const CudaOpArgs &inputs,
              const CudaOpArgs &outputs,
              const std::shared_ptr<CudaTensorStorageMemory> &mem,
              const size_t elements, const char *name)
      : CudaOperation(name)
      , m_inputs(inputs)
      , m_outputs(outputs)
      , m_mem(mem)
      , m_iter(0)
      , m_ctx(p.m_ctx)
      , m_elements(elements)
    {
    }

    CudaOpArgs listInputs() const override { return m_inputs; }

    CudaOpArgs listOutputs() const override { return m_outputs; }
};

struct AdamF32 : public Optimizer {
    AdamF32(CudaProgram &p, const CudaOpArgs &inputs, const CudaOpArgs &outputs,
            const std::shared_ptr<CudaTensorStorageMemory> &mem,
            const size_t elements)
      : Optimizer(p, inputs, outputs, mem, elements, "adam-f32")
      , m_mvec(sizeof(float) * elements / p.m_ctx->m_engine->m_nodes)
      , m_vvec(sizeof(float) * elements / p.m_ctx->m_engine->m_nodes)
    {
        m_weights = (float *)m_mem->m_mem;
        m_gradients = m_weights + elements;
    }

    const char *exec(CudaProgram &p, long batch) override
    {
        const int i = ++m_iter;
        const float b1t = 1.0 / (1.0 - pow(ADAM_B1, i));
        const float b2t = 1.0 / (1.0 - pow(ADAM_B2, i));

        const auto &engine = *p.m_ctx->m_engine;
        const int rank = p.m_ctx->m_nccl_rank;

        const size_t local_elements = m_elements / engine.m_nodes;
        const size_t offset = local_elements * rank;

#ifdef HAVE_NCCL
        if(engine.m_nodes > 1) {
            ncclReduceScatter(gradients, gradients + offset, elements,
                              ncclFloat32, ncclSum, engine.m_nccl_comms[rank],
                              p.m_ctx->m_stream);
        }
#endif
        // clang-format off
        adam_float(local_elements,
                   m_weights + offset,
                   m_gradients + offset,
                   (float *)m_mvec.m_mem,
                   (float *)m_vvec.m_mem,
                   b1t, b2t,
                   p.m_pc.learning_rate, p.m_pc.l2_lambda, p.m_aux,
                   p.m_ctx->m_stream, p.m_ctx->m_num_sm);
        // clang-format on

#ifdef HAVE_NCCL
        if(engine.m_nodes > 1) {
            ncclAllGather(weights + offset, weights, elements, ncclFloat32,
                          engine.m_nccl_comms[rank], p.m_ctx->m_stream);
        }
#endif
        return NULL;
    }

    float *m_weights;
    float *m_gradients;

    CudaTensorStorageMemory m_mvec;
    CudaTensorStorageMemory m_vvec;
};

struct AdamMixed : public Optimizer {
    AdamMixed(CudaProgram &p, const CudaOpArgs &inputs,
              const CudaOpArgs &outputs,
              const std::shared_ptr<CudaTensorStorageMemory> &mem,
              const size_t elements)
      : Optimizer(p, inputs, outputs, mem, elements, "adam-mixed")
      , m_mvec(sizeof(float) * elements / p.m_ctx->m_engine->m_nodes)
      , m_vvec(sizeof(float) * elements / p.m_ctx->m_engine->m_nodes)
      , m_cvec(sizeof(float) * elements / p.m_ctx->m_engine->m_nodes)
    {
        const auto &engine = *p.m_ctx->m_engine;
        const int rank = p.m_ctx->m_nccl_rank;

        __half *hmem = (__half *)m_mem->m_mem;
        m_weights = hmem;
        m_gradients = hmem + m_elements;

        const size_t local_elements = m_elements / engine.m_nodes;
        const size_t offset = local_elements * rank;

        float *cvec = (float *)m_cvec.m_mem;
        for(size_t i = 0; i < elements; i++) {
            cvec[i] = m_weights[i + offset];
        }
    }

    const char *exec(CudaProgram &p, long batch) override
    {
        const int i = ++m_iter;
        const float b1t = 1.0 / (1.0 - pow(ADAM_B1, i));
        const float b2t = 1.0 / (1.0 - pow(ADAM_B2, i));

        const auto &engine = *p.m_ctx->m_engine;
        const int rank = p.m_ctx->m_nccl_rank;

        const size_t local_elements = m_elements / engine.m_nodes;
        const size_t offset = local_elements * rank;

#ifdef HAVE_NCCL
        if(engine.m_nodes > 1) {
            ncclReduceScatter(gradients, gradients + offset, elements,
                              ncclFloat16, ncclSum, engine.m_nccl_comms[rank],
                              p.m_ctx->m_stream);
        }
#endif

        // clang-format off
        adam_mixed(m_elements,
                   1.0f / p.m_mp_scaling,
                   m_weights + offset,
                   m_gradients + offset,
                   (float *)m_mvec.m_mem,
                   (float *)m_vvec.m_mem,
                   (float *)m_cvec.m_mem,
                   b1t, b2t, p.m_pc.learning_rate,
                   p.m_pc.l2_lambda, p.m_aux, p.m_ctx->m_stream,
                   p.m_ctx->m_num_sm);
        // clang-format on

#ifdef HAVE_NCCL
        if(engine.m_nodes > 1) {
            ncclAllGather(weights + offset, weights, elements, ncclFloat16,
                          engine.m_nccl_comms[rank], p.m_ctx->m_stream);
        }
#endif

        return NULL;
    }

    __half *m_weights;
    __half *m_gradients;

    CudaTensorStorageMemory m_mvec;
    CudaTensorStorageMemory m_vvec;
    CudaTensorStorageMemory m_cvec;
};

CudaOp
CudaProgram::create_optimizer(Tensor::DataType dt)
{
    const size_t align = 16;

    size_t total_elements = 0;

    for(auto it : m_updates) {
        auto &w = it.first;
        auto &g = it.second;

        if(w->m_data_type != dt)
            continue;

        // Get rid of this dep
        assert(!w->m_storage->m_memory && !g->m_storage->m_memory);

        assert(w->m_dims == g->m_dims);
        assert(w->m_data_type == g->m_data_type);

        size_t elements = w->m_dims.elements();
        size_t rounded_elements = (elements + align - 1) & ~(align - 1);
        total_elements += rounded_elements;
    }

    if(total_elements == 0)
        return nullptr;

    const size_t esize = Tensor::DataTypeSize(dt);
    const size_t size = total_elements * 2 * esize;
    auto mem = std::make_shared<CudaTensorStorageMemory>(size);

    char *wbase = (char *)mem->m_mem;
    char *gbase = wbase + total_elements * esize;

    total_elements = 0;

    CudaOpArgs inputs, outputs;

    for(auto it : m_updates) {
        auto &w = it.first;
        auto &g = it.second;

        if(w->m_data_type != dt)
            continue;

        const size_t elements = w->m_dims.elements();
        const size_t rounded_elements = (elements + align - 1) & ~(align - 1);

        w->m_storage->m_memory = mem;
        g->m_storage->m_memory = mem;

        w->m_storage->m_data = wbase + total_elements * esize;
        g->m_storage->m_data = gbase + total_elements * esize;

        inputs.push_back({"g", g});
        inputs.push_back({"w", w});
        outputs.push_back({"w", w});

        auto src = m_ctx->m_deferred_copy[w];
        w->copyFromLocked(*src, 0);
        m_ctx->m_deferred_copy.erase(w);

        total_elements += rounded_elements;
    }

    switch(dt) {
    case Tensor::DataType::FLOAT:
        return std::make_shared<AdamF32>(*this, inputs, outputs, mem,
                                         total_elements);
    case Tensor::DataType::HALF:
        return std::make_shared<AdamMixed>(*this, inputs, outputs, mem,
                                           total_elements);
    default:
        abort();
    }
}

CudaOps
CudaProgram::create_optimizers()
{
    CudaOps ops;

    auto o = create_optimizer(Tensor::DataType::FLOAT);
    if(o)
        ops.push_back(o);

    o = create_optimizer(Tensor::DataType::HALF);
    if(o) {
        m_mp_enabled = true;
        ops.push_back(o);
    }

    return ops;
}

}  // namespace saga
