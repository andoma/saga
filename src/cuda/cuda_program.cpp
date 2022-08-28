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
#include <map>

#include "saga.hpp"
#include "tensor.hpp"
#include "engine.hpp"

#include "cuda_common.hpp"
#include "cuda_tensor.hpp"
#include "cuda_analysis.hpp"

namespace saga {

void
CudaProgramUnit::fwd(const std::shared_ptr<CudaOperation> &op)
{
    m_fwd_operations.push_back(op);
}

void
CudaProgramUnit::bwd(const std::shared_ptr<CudaOperation> &op)
{
    m_bwd_operations.insert(m_bwd_operations.begin(), op);
}

void
CudaProgramUnit::tail(const std::shared_ptr<CudaOperation> &op)
{
    m_tail_operations.push_back(op);
}

float
CudaProgram::upd(const std::shared_ptr<CudaTensor> &weights,
                 const std::shared_ptr<CudaTensor> &gradient)
{
    auto &p = m_updates[weights];
    if(p) {
        assert(p.get() == gradient.get());
        return 1.0f;
    } else {
        p = gradient;
        return 0;
    }
}

cudnnTensorFormat_t
CudaProgram::tensorFormat(Tensor::DataType data_type)
{
    switch(m_pc.tensor_layout) {
    case TensorLayout::Auto:

        switch(data_type) {
        case Tensor::DataType::U8:
        case Tensor::DataType::HALF:
            if(m_ctx->m_tensor_cores)
                return CUDNN_TENSOR_NHWC;
            // FALLTHRU
        default:
            return CUDNN_TENSOR_NCHW;
        }

    case TensorLayout::NHWC:
        return CUDNN_TENSOR_NHWC;

    case TensorLayout::NCHW:
        return CUDNN_TENSOR_NCHW;
    }
    abort();
}

std::shared_ptr<CudaTensor>
CudaProgram::lower_tensor(const CudaProgramUnit &pu,
                          std::shared_ptr<Tensor> src, size_t minimum_rank)
{
    if(src == nullptr)
        return nullptr;

    auto it = m_ctx->m_tensors.find(src);
    if(it != m_ctx->m_tensors.end()) {
        return it->second;
    }

    Dims dims = src->dims_;

    if(minimum_rank) {
        while(dims.size() < minimum_rank) dims.insert(dims.begin(), 1);
    }

    dims = dims.batch(pu.m_batch_size);

    auto t = std::make_shared<CudaTensor>(
        src->data_type_, dims, tensorFormat(*src), m_ctx, src->name_);

    m_ctx->m_deferred_copy[t] = src;
    m_ctx->m_tensors[src] = t;
    return t;
}

std::shared_ptr<CudaTensor>
CudaProgram::lower_tensor(std::shared_ptr<Tensor> src,
                          const CudaTensor &blueprint)
{
    if(src == nullptr)
        return nullptr;

    auto it = m_ctx->m_tensors.find(src);
    if(it != m_ctx->m_tensors.end()) {
        return it->second;
    }

    auto t = std::make_shared<CudaTensor>(src->data_type_, blueprint);
    m_ctx->m_deferred_copy[t] = src;
    m_ctx->m_tensors[src] = t;
    return t;
}

std::shared_ptr<CudaTensor>
CudaProgram::lower_tensor(const CudaProgramUnit &pu,
                          std::shared_ptr<Tensor> src,
                          cudnnTensorFormat_t tensor_format)
{
    if(src == nullptr)
        return nullptr;

    auto it = m_ctx->m_tensors.find(src);
    if(it != m_ctx->m_tensors.end()) {
        return it->second;
    }

    auto t = std::make_shared<CudaTensor>(src->data_type_,
                                          src->dims_.batch(pu.m_batch_size),
                                          tensor_format, m_ctx, src->name_);

    m_ctx->m_deferred_copy[t] = src;
    m_ctx->m_tensors[src] = t;
    return t;
}

std::shared_ptr<CudaTensor>
CudaProgram::lower_grad(const CudaProgramUnit &pu, std::shared_ptr<Tensor> src,
                        size_t minimum_rank)
{
    if(src == nullptr)
        return nullptr;
    src = src->grad();
    assert(src.get());
    return lower_tensor(pu, src, minimum_rank);
}

std::shared_ptr<CudaTensor>
CudaProgram::lower_grad(const CudaProgramUnit &pu, std::shared_ptr<Tensor> src,
                        cudnnTensorFormat_t format)
{
    if(src == nullptr)
        return nullptr;
    src = src->grad();
    assert(src.get());
    return lower_tensor(pu, src, format);
}

void
CudaProgram::setupTensorStorage(std::shared_ptr<CudaMemoryLayout> cml)
{
    if(!cml)
        return;

    void *base = m_ctx->m_tensor_mem.ptr();
    for(const auto &kv : cml->table_) {
        void *ptr = (void *)((char *)base + kv.second);
        //        printf("%p: Tensor %s gets %p\n", this,
        //        kv.first->name().c_str(), ptr);
        kv.first->setTmpMem(ptr);
    }
}

void
CudaProgram::detect_anomaly(
    const CudaOperation &op,
    const std::vector<std::shared_ptr<CudaTensor>> &tensors, const char *what)
{
    for(const auto &t : tensors) {
        t->detect_anomaly(&m_aux->anomaly);
        chkCuda(cudaStreamSynchronize(m_ctx->m_stream));
        if(m_aux->anomaly) {
            fprintf(stderr, "\n\n *** Anomaly in tensor %s (%s)\n",
                    t->storage_name().c_str(), what);

            op.dump(stderr, 1);

            for(const auto &t : op.getInputs()) {
                t->print_anomaly("I");
                t->printStats("I");
            }
            for(const auto &t : op.getOutputs()) {
                t->print_anomaly("O");
                t->printStats("O");
            }

            printf("Pre invalidate: ");
            for(auto &t : op.m_pre_invalidate) {
                for(const auto &id : t->m_idmap) {
                    printf("T%d ", id.second);
                }
            }
            printf("\n");

            printf("Post invalidate: ");
            for(auto &t : op.m_post_invalidate) {
                for(const auto &id : t->m_idmap) {
                    printf("T%d ", id.second);
                }
            }
            printf("\n");

            FILE *fp = fopen("anomaly.txt", "w");
            if(fp != NULL) {
                printf("Dumping program to anomaly.txt\n");
                dump(fp, true);
                fclose(fp);
            }
            abort();
        }
    }
}

bool
CudaProgram::runOps(const CudaOps &ops, long batch, bool anomaly_detect)
{
    for(const auto &op : ops) {
        if(anomaly_detect) {
            for(auto &t : op->m_pre_invalidate) {
                t->invalidate();
            }
            detect_anomaly(*op, op->getInputs(), "input");
        }

        const char *err = op->exec(*this, batch);
        if(err) {
            fprintf(stderr, "\nOp %s failed: %s\n", op->name().c_str(), err);
            op->dump(stderr, true);
            return false;
        }

        if(anomaly_detect) {
            detect_anomaly(*op, op->getOutputs(), "output");
            for(auto &t : op->m_post_invalidate) {
                t->invalidate();
            }
        }
    }
    return true;
}

void
CudaProgram::prep(long batches, long batch_offset)
{
    finalize();

    m_pc.ui->updateCell(m_ui_row, 1, UI::Align::LEFT, "Running");

    run_batched_tensor_callbacks(m_pc.pre_ops, 0 + batch_offset, Phase::PRE);

    flipDoubleBufferedTensors();

    chkCuda(cudaStreamSynchronize(m_ctx->m_stream));

    setupTensorStorage(m_memory_layout);
    m_total_samples = 0;
    m_epoch_start = Now();
}

ExecResult
CudaProgram::step(long batch, long batches, long batch_offset)
{
    if(!runOps(m_ops, batch, m_pc.anomaly_detect)) {
        return ExecResult::ERROR;
    }

    if(batch > 0) {
        run_batched_tensor_callbacks(m_pc.post_ops, batch - 1 + batch_offset,
                                     Phase::POST);
    }

    if(batch < batches - 1) {
        run_batched_tensor_callbacks(m_pc.pre_ops, batch + 1 + batch_offset,
                                     Phase::PRE);
    }

    flipDoubleBufferedTensors();

    for(const auto &u : m_units) {
        m_total_samples += u.m_batch_size;
    }

    chkCuda(cudaStreamSynchronize(m_ctx->m_stream));

    if(m_mp_enabled) {
        if(m_aux->inf || m_aux->nan) {
            m_mp_scaling *= 0.5;
        } else if(m_aux->range > 1) {
            m_mp_scaling *= 0.9;
        } else {
            m_mp_scaling *= 1.02;
        }

        m_aux->range = 0;
        m_aux->inf = 0;
        m_aux->nan = 0;
    }

    float sps = m_total_samples / (1e-6 * (Now() - m_epoch_start));

    m_pc.ui->updateCell(m_ui_row, 7, UI::Align::LEFT, "%d (%.1f/s)",
                        m_total_samples, sps);
    m_pc.ui->updateCell(m_ui_row, 5, UI::Align::LEFT, "%ld / %ld", batch,
                        batches);
    if(m_mp_enabled)
        m_pc.ui->updateCell(m_ui_row, 9, UI::Align::LEFT, "%.1e", m_mp_scaling);

    return ExecResult::OK;
}

void
CudaProgram::post(long batches, long batch_offset)
{
    m_pc.ui->updateCell(m_ui_row, 1, UI::Align::LEFT, "Paused");
    run_batched_tensor_callbacks(m_pc.post_ops, batches - 1 + batch_offset,
                                 Phase::POST);
}

ExecResult
CudaProgram::run(long batches, StopCheck stop_check, long batch_offset)
{
    if(batches == 0)
        return ExecResult::OK;

    if(stop_check && stop_check())
        return ExecResult::STOPPED;

    prep(batches, batch_offset);

    for(long i = 0; i < batches; i++) {
        ExecResult r = step(i, batches, batch_offset);
        if(r != ExecResult::OK)
            return r;

        if(stop_check && stop_check()) {
            return ExecResult::STOPPED;
        }

        m_ctx->updateGpuStats();
    }

    post(batches, batch_offset);
    return ExecResult::OK;
}

void
CudaProgram::dump(FILE *output, bool detailed) const
{
    fprintf(output, "\n%zd ops\n", m_ops.size());
    int index = 0;
    for(const auto &op : m_ops) {
        fprintf(output, "#%3d: ", index);
        op->dump(output, detailed);
        index++;
    }
}

void
CudaProgram::debug(bool on)
{
    m_debug = on;
}

void
CudaProgram::addBatchedTensors(const ProgramSource &ps)
{
    for(const auto &bt : ps.batched_tensors) {
        auto src = bt.first;

        auto &p = m_batched_tensors[src];

        if(!p.first) {
            auto dims = src->dims_.batch(ps.batch_size);
            auto fmt = tensorFormat(*src);
            auto s = std::make_shared<CudaTensorStorageDoubleBuffered>(
                src->data_type_, dims, fmt, m_ctx);
            auto t = std::make_shared<CudaTensor>(s, dims, fmt);

            m_flips.push_back(s);
            t->copyFromLocked(*src);
            m_ctx->m_tensors[src] = t;
            p.first = s;
        }

        if(!(p.second & Phase::POST) && !!(bt.second & Phase::POST)) {
            m_ctx->m_exported_storage.push_back(p.first);
        }
        p.second = p.second | bt.second;
    }
}

void
CudaProgram::finalize()
{
    if(m_finalized) {
        return;
    }
    m_finalized = true;

    for(const auto &[to, from] : m_ctx->m_deferred_copy) {
        to->copyFromLocked(*from, 0);
    }
    m_ctx->m_deferred_copy.clear();

    for(auto &pu : m_units) {
        m_ops.insert(m_ops.end(), pu.m_fwd_operations.begin(),
                     pu.m_fwd_operations.end());
        m_ops.insert(m_ops.end(), pu.m_bwd_operations.begin(),
                     pu.m_bwd_operations.end());

        pu.m_fwd_operations.clear();
        pu.m_bwd_operations.clear();
    }

    m_ops.insert(m_ops.end(), m_opt.begin(), m_opt.end());
    m_opt.clear();

    if(m_ops.size()) {
        m_ops = reduceLiveranges(m_ops, m_ctx->m_exported_storage);

        m_memory_layout =
            memoryLayout(m_ops, m_ctx->m_exported_storage, m_pc.anomaly_detect);

        m_ctx->m_tensor_mem.request(m_memory_layout->size_);
    }

    m_ctx->m_tensor_mem.alloc();

    m_ctx->m_workspace.alloc();

    if(m_mp_enabled) {
        m_pc.ui->updateCell(m_ui_row, 8, UI::Align::RIGHT, "MPS:");
    }
}

bool
CudaProgram::dumpGraphFromOps(const char *path, const CudaOps &ops)
{
    FILE *fp = fopen(path, "w");
    if(fp == NULL) {
        perror("fopen");
        return false;
    }

    std::unordered_set<std::shared_ptr<CudaTensorStorage>> storage;

    fprintf(fp, "digraph CudaOps {\n");

    int i = 0;
    for(auto &op : ops) {
        fprintf(fp, "node [shape=circle,label=\"%s\"]; O%d;\n",
                op->name().c_str(), i);
        i++;

        for(auto const &t : op->getInputs()) {
            storage.insert(t->m_storage);
        }
        for(auto const &t : op->getOutputs()) {
            storage.insert(t->m_storage);
        }
    }

    for(auto const &s : storage) {
        const auto name = s->name();
        fprintf(fp, "node [shape=box,label=\"%s\"]; %s;\n", name.c_str(),
                name.c_str());
    }
    fprintf(fp, "\n");

    i = 0;
    for(auto &op : ops) {
        for(auto const &t : op->getInputs()) {
            const auto name = t->storage_name();
            fprintf(fp, "%s->O%d;\n", name.c_str(), i);
        }
        for(auto const &t : op->getOutputs()) {
            const auto name = t->storage_name();
            fprintf(fp, "O%d->%s;\n", i, name.c_str());
        }
        i++;
    }

    fprintf(fp, "overlap=false\n");

    fprintf(fp, "}\n");
    fclose(fp);
    return true;
}

bool
CudaProgram::dumpGraph(const char *path)
{
    return dumpGraphFromOps(path, m_ops);
}

}  // namespace saga
