/*
 * Copyright (c) 2020, Andreas Smas
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

#include <string.h>
#include <sstream>
#include "dnnl_tensor.h"
#include "dnnl_debug.h"

#include "tensor.h"

namespace saga {

class DnnlTensorStorage : public TensorStorage {
public:
    DnnlTensorStorage(Tensor::DataType data_type, size_t size,
                      const std::shared_ptr<DnnlContext> &ctx)
      : TensorStorage(data_type), ctx_(ctx)
    {
        data_ = valloc(size);
        memset(data_, 0, size);
    }

    ~DnnlTensorStorage() { free(data_); }

    const std::shared_ptr<DnnlContext> ctx_;
};

class DnnlTensorAccess : public TensorAccess {
public:
    DnnlTensorAccess(std::shared_ptr<DnnlTensorStorage> storage,
                     const dnnl_memory_desc_t *desc)
      : storage_(storage), desc_(*desc), sync_(false)
    {
        //    storage_->ctx_->mutex_.lock();
    }

    ~DnnlTensorAccess()
    {
        //    storage_->ctx_->mutex_.unlock();
    }

    Dims strides() { return {}; }

    void *data() { return nullptr; }

    int64_t offsetForElement(const Dims &element) const
    {
        int64_t pos[desc_.ndims];
        size_t i = 0;
        for(; i < element.size(); i++) pos[i] = element[i];

        for(; i < (size_t)desc_.ndims; i++) pos[i] = 0;

        dnnl_dim_t offset = desc_.offset0;
        const auto &blk = desc_.format_desc.blocking;

        if(blk.inner_nblks > 0) {
            int64_t blk_stride = 1;
            for(int iblk = blk.inner_nblks - 1; iblk >= 0; --iblk) {
                const int d = blk.inner_idxs[iblk];
                int64_t p;
                if(pos[d] <= INT32_MAX) {
                    p = (int32_t)pos[d] % (int32_t)blk.inner_blks[iblk];
                    pos[d] = (int32_t)pos[d] / (int32_t)blk.inner_blks[iblk];
                } else {
                    p = pos[d] % blk.inner_blks[iblk];
                    pos[d] = pos[d] / blk.inner_blks[iblk];
                }
                offset += p * blk_stride;
                blk_stride *= blk.inner_blks[iblk];
            }
        }

        for(int d = 0; d < desc_.ndims; d++) {
            offset += pos[d] * blk.strides[d];
        }
        return offset;
    }

    virtual double get(const Dims &element)
    {
        if(!sync_) {
            chkDNNL(dnnl_stream_wait(storage_->ctx_->stream_));
            sync_ = true;
        }
        return storage_->get(offsetForElement(element));
    };

    virtual void set(const Dims &element, double value)
    {
        if(!sync_) {
            chkDNNL(dnnl_stream_wait(storage_->ctx_->stream_));
            sync_ = true;
        }

        storage_->set(offsetForElement(element), value);
    }

    virtual void copyBytesFrom(const Dims &element, const void *data,
                               size_t size)
    {
        fprintf(stderr, "%s not implemented\n", __FUNCTION__);
        abort();
    }

    const std::shared_ptr<DnnlTensorStorage> storage_;
    dnnl_memory_desc_t desc_;
    bool sync_;
};

#if 0
static dnnl_data_type_t
dnnlDataType_from_dataType(Tensor::DataType data_type)
{
  switch(data_type) {
  case Tensor::DataType::FLOAT:
    return dnnl_f32;
  case Tensor::DataType::HALF:
    return dnnl_f16;
  case Tensor::DataType::U8:
    return dnnl_u8;
  case Tensor::DataType::I32:
    return dnnl_s32;
  default:
    fprintf(stderr, "Unsupported data_type %d for dnnl tensor\n",
            (int)data_type);
    abort();
  }
}

static void
dnnl_dims_from_dims(dnnl_dims_t dims, const Dims &src)
{
  assert(src.size() <= DNNL_MAX_NDIMS);
  for(int i = 0; i < src.size(); i++)
    dims[i] = src[i];
}
#endif

static Tensor::DataType
dmd_to_datatype(const dnnl_memory_desc_t *desc)
{
    switch(desc->data_type) {
    case dnnl_f16:
        return Tensor::DataType::HALF;
    case dnnl_f32:
        return Tensor::DataType::FLOAT;
    case dnnl_s32:
        return Tensor::DataType::I32;
    case dnnl_u8:
        return Tensor::DataType::U8;
    default:
        fprintf(stderr, "%s: Can't handle data_type %d\n", __FUNCTION__,
                desc->data_type);
        abort();
    }
}

static Dims
dmd_to_dims(const dnnl_memory_desc_t *desc)
{
    Dims r;
    r.reserve(desc->ndims);
    for(int i = 0; i < desc->ndims; i++) r.push_back(desc->dims[i]);
    return r;
}

DnnlTensor::DnnlTensor(const dnnl_memory_desc_t *desc,
                       const std::shared_ptr<DnnlContext> &ctx,
                       const std::optional<const std::string> &name)
  : Tensor(dmd_to_datatype(desc), dmd_to_dims(desc), name), desc_(*desc)
{
    const size_t byte_size = dnnl_memory_desc_get_size(desc);
    storage_ = std::make_shared<DnnlTensorStorage>(data_type_, byte_size, ctx);
    chkDNNL(
        dnnl_memory_create(&memory_, &desc_, ctx->engine_, storage_->data_));
}

DnnlTensor::~DnnlTensor() { chkDNNL(dnnl_memory_destroy(memory_)); }

std::string
DnnlTensor::info() const
{
    std::stringstream ss;
    ss << Tensor::info();

    char tmp1[512];
    char tmp2[512];
    dnnl_md2fmt_str(tmp1, sizeof(tmp1), &desc_);
    dnnl_md2dim_str(tmp2, sizeof(tmp2), &desc_);

    ss << " " << tmp1 << " [" << tmp2 << "]";

    return ss.str();
}

std::unique_ptr<TensorAccess>
DnnlTensor::access()
{
    return std::make_unique<DnnlTensorAccess>(storage_, &desc_);
}

std::shared_ptr<Tensor>
DnnlTensor::slice(const Dims &offset, const Dims &size)
{
    fprintf(stderr, "%s not implemented\n", __FUNCTION__);
    abort();
    return nullptr;
}

void
DnnlTensor::copyFromLocked(Tensor &t)
{
    //  printf("Copy %s\n  To %s\n", t.info().c_str(), info().c_str());
    auto ta = t.access();
    if(ta == NULL)
        return;

    std::shared_ptr<Tensor> tmp_tensor;

    void *src_data = ta->data();

    if(src_data == NULL) {
        ta.reset();
        tmp_tensor = makeCPUTensor(t.data_type_, t.dims_);
        tmp_tensor->copyFrom(t);
        ta = tmp_tensor->access();
        src_data = ta->data();
    }

    auto src_strides_vec = ta->strides().i64();
    auto src_dims_vec = t.dims_.i64();

    int64_t *src_strides = &src_strides_vec[0];
    int64_t *src_dims = &src_dims_vec[0];

    size_t src_rank = t.dims_.size();
    while(src_rank > dims_.size() && src_dims[0] == 1) {
        src_dims++;
        src_strides++;
        src_rank--;
    }

    assert(t.data_type_ == Tensor::DataType::FLOAT);
    dnnl_memory_desc_t src_desc;
    chkDNNL(dnnl_memory_desc_init_by_strides(&src_desc, src_rank, src_dims,
                                             dnnl_f32, src_strides));

    dnnl_memory_t src_memory;
    const DnnlContext &ctx = *storage_->ctx_;
    chkDNNL(dnnl_memory_create(&src_memory, &src_desc, ctx.engine_, src_data));

    dnnl_primitive_desc_t pd;

    chkDNNL(dnnl_reorder_primitive_desc_create(&pd, &src_desc, ctx.engine_,
                                               &desc_, ctx.engine_, NULL));

    dnnl_primitive_t prim;
    chkDNNL(dnnl_primitive_create(&prim, pd));

    dnnl_exec_arg_t args[2] = {{DNNL_ARG_SRC, src_memory},
                               {DNNL_ARG_DST, memory_}};

    chkDNNL(dnnl_primitive_execute(prim, ctx.stream_, 2, args));
    chkDNNL(dnnl_primitive_desc_destroy(pd));
    chkDNNL(dnnl_primitive_destroy(prim));
    chkDNNL(dnnl_memory_destroy(src_memory));
    chkDNNL(dnnl_stream_wait(ctx.stream_));
}

void
DnnlTensor::copyFrom(Tensor &t)
{
    // XXX: Add locking
    copyFromLocked(t);
}

}  // namespace saga
