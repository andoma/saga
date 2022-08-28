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
    std::vector<std::shared_ptr<CudaTensor>> m_inputs;
    std::vector<std::shared_ptr<CudaTensor>> m_outputs;
    const std::shared_ptr<CudaTensorStorageMemory> m_mem;
    int m_iter{0};
    const std::shared_ptr<CudaContext> m_ctx;
    const size_t m_elements;

    Optimizer(CudaProgram &p, std::vector<std::shared_ptr<CudaTensor>> inputs,
              std::vector<std::shared_ptr<CudaTensor>> outputs,
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

    std::vector<std::shared_ptr<CudaTensor>> getInputs() const override
    {
        return m_inputs;
    }

    std::vector<std::shared_ptr<CudaTensor>> getOutputs() const override
    {
        return m_outputs;
    }
};

struct AdamF32 : public Optimizer {
    AdamF32(CudaProgram &p, std::vector<std::shared_ptr<CudaTensor>> inputs,
            std::vector<std::shared_ptr<CudaTensor>> outputs,
            const std::shared_ptr<CudaTensorStorageMemory> &mem,
            const size_t elements)
      : Optimizer(p, inputs, outputs, mem, elements, "adam-f32")
    {
    }

    const char *exec(CudaProgram &p, long batch) override
    {
        const int i = ++m_iter;
        const float b1t = 1.0 / (1.0 - pow(ADAM_B1, i));
        const float b2t = 1.0 / (1.0 - pow(ADAM_B2, i));

        float *mem = (float *)m_mem->m_mem;
        adam_float(m_elements, mem, mem + m_elements, mem + m_elements * 2,
                   mem + m_elements * 3, b1t, b2t, p.m_pc.learning_rate,
                   p.m_pc.l2_lambda, p.m_aux, p.m_ctx->m_stream,
                   p.m_ctx->m_num_sm);
        return NULL;
    }
};

struct AdamMixed : public Optimizer {
    AdamMixed(CudaProgram &p, std::vector<std::shared_ptr<CudaTensor>> inputs,
              std::vector<std::shared_ptr<CudaTensor>> outputs,
              const std::shared_ptr<CudaTensorStorageMemory> &mem,
              const size_t elements)
      : Optimizer(p, inputs, outputs, mem, elements, "adam-mixed")
    {
        __half *hmem = (__half *)m_mem->m_mem;
        float *fmem = (float *)m_mem->m_mem;

        m_weights = hmem;
        m_gradients = hmem + m_elements;
        m_mvec = fmem + m_elements * 1;
        m_vvec = fmem + m_elements * 2;
        m_cvec = fmem + m_elements * 3;

        for(size_t i = 0; i < elements; i++) {
            m_cvec[i] = m_weights[i];
        }
    }

    const char *exec(CudaProgram &p, long batch) override
    {
        const int i = ++m_iter;
        const float b1t = 1.0 / (1.0 - pow(ADAM_B1, i));
        const float b2t = 1.0 / (1.0 - pow(ADAM_B2, i));

        adam_mixed(m_elements, 1.0f / p.m_mp_scaling, m_weights, m_gradients,
                   m_mvec, m_vvec, m_cvec, b1t, b2t, p.m_pc.learning_rate,
                   p.m_pc.l2_lambda, p.m_aux, p.m_ctx->m_stream,
                   p.m_ctx->m_num_sm);
        return NULL;
    }

    __half *m_weights;
    __half *m_gradients;
    float *m_cvec;
    float *m_mvec;
    float *m_vvec;
};

CudaOp
CudaProgram::create_optimizer(Tensor::DataType dt)
{
    const size_t align = 16;

    size_t total_elements = 0;

    for(auto it : m_updates) {
        auto &w = it.first;
        auto &g = it.second;

        if(w->data_type_ != dt)
            continue;

        // Get rid of this dep
        assert(!w->m_storage->m_memory && !g->m_storage->m_memory);

        assert(w->dims_ == g->dims_);
        assert(w->data_type_ == g->data_type_);

        size_t elements = w->dims_.elements();
        size_t rounded_elements = (elements + align - 1) & ~(align - 1);
        total_elements += rounded_elements;
    }

    if(total_elements == 0)
        return nullptr;

    size_t size;
    switch(dt) {
    case Tensor::DataType::FLOAT:
        size = total_elements * sizeof(float) * 4;
        break;
    case Tensor::DataType::HALF:
        size = total_elements * (sizeof(uint16_t) * 2 + sizeof(float) * 3);
        break;
    default:
        abort();
    }

    auto mem = std::make_shared<CudaTensorStorageMemory>(size);

    char *wbase = (char *)mem->m_mem;
    char *gbase = wbase + Tensor::DataTypeSize(dt) * total_elements;

    total_elements = 0;

    std::vector<std::shared_ptr<CudaTensor>> inputs;
    std::vector<std::shared_ptr<CudaTensor>> outputs;

    for(auto it : m_updates) {
        auto &w = it.first;
        auto &g = it.second;

        if(w->data_type_ != dt)
            continue;

        size_t elements = w->dims_.elements();
        size_t rounded_elements = (elements + align - 1) & ~(align - 1);

        w->m_storage->m_memory = mem;
        g->m_storage->m_memory = mem;
        switch(dt) {
        case Tensor::DataType::FLOAT:
            w->m_storage->m_data = wbase + total_elements * sizeof(float);
            g->m_storage->m_data = gbase + total_elements * sizeof(float);
            break;
        case Tensor::DataType::HALF:
            w->m_storage->m_data = wbase + total_elements * sizeof(__half);
            g->m_storage->m_data = gbase + total_elements * sizeof(__half);
            break;
        default:
            abort();
        }

        inputs.push_back(g);
        inputs.push_back(w);
        outputs.push_back(w);

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
