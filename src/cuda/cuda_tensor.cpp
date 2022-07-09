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

CudaTensorStorage::CudaTensorStorage(Tensor::DataType data_type, size_t size,
                                     const std::shared_ptr<CudaContext> &ctx)
  : TensorStorage(data_type)
  , m_ctx(ctx)
  , m_size(size)
  , m_element_size(Tensor::DataTypeSize(data_type))
  , m_id(++ctx->m_tensor_storage_id_gen)
{
}

CudaTensorStorage::~CudaTensorStorage()
{
    if(m_mem)
        assert(m_mem == m_data);
    chkCuda(cudaFree(m_mem));
}

void
CudaTensorStorage::alloc()
{
    if(m_mem)
        return;
    chkCuda(cudaMallocManaged(&m_mem, m_size, cudaMemAttachGlobal));
    chkCuda(cudaMemset(m_mem, 0, m_size));
    m_data = m_mem;
}

void
CudaTensorStorage::setTmpMem(void *p)
{
    if(m_mem)
        return;
    m_data = p;
}

void *
CudaTensorStorage::deviceMem(int64_t offset)
{
    return (void *)((char *)m_data + offset * m_element_size);
}

class CudaTensorAccess : public TensorAccess {
public:
    CudaTensorAccess(std::shared_ptr<CudaTensorStorage> storage,
                     cudnnTensorDescriptor_t desc, int64_t offset)
      : m_storage(storage), m_offset(offset), m_sync(false)
    {
        const int max_rank = 8;
        int dims[max_rank];
        int strides[max_rank];
        int rank;
        cudnnDataType_t data_type;

        chkCUDNN(cudnnGetTensorNdDescriptor(desc, max_rank, &data_type, &rank,
                                            dims, strides));

        for(int i = 0; i < rank; i++) {
            m_strides.push_back(strides[i]);
        }

        m_storage->m_ctx->m_mutex.lock();
    }

    ~CudaTensorAccess() { m_storage->m_ctx->m_mutex.unlock(); }

    Dims strides() { return m_strides; }

    void *data() { return m_storage->data(); }

    int64_t offsetForElement(const Dims &element) const
    {
        size_t offset = m_offset;
        for(size_t i = 0; i < element.size() && i < m_strides.size(); i++) {
            offset += element[i] * m_strides[i];
        }
        return offset;
    }

    virtual double get(const Dims &element)
    {
        if(!m_sync) {
            cudaStreamSynchronize(m_storage->m_ctx->m_stream);
            m_sync = true;
        }
        return m_storage->get(offsetForElement(element));
    };

    void *getAddr(const Dims &element) override
    {
        size_t off = offsetForElement(element) * m_storage->m_element_size;
        return (void *)((char *)m_storage->data() + off);
    };

    virtual void set(const Dims &element, double value)
    {
        m_storage->set(offsetForElement(element), value);
    }

    virtual void copyBytesFrom(const Dims &element, const void *data,
                               size_t size)
    {
        const size_t o = offsetForElement(element) * m_storage->m_element_size;
        char *dst = (char *)m_storage->data();
        cudaMemcpy(dst + o, data, size, cudaMemcpyHostToDevice);
    }

    Dims m_strides;
    const std::shared_ptr<CudaTensorStorage> m_storage;
    const int64_t m_offset;
    bool m_sync;
};

cudnnDataType_t
cudnnDataType_from_dataType(Tensor::DataType data_type)
{
    switch(data_type) {
    case Tensor::DataType::FLOAT:
        return CUDNN_DATA_FLOAT;
    case Tensor::DataType::HALF:
        return CUDNN_DATA_HALF;
    case Tensor::DataType::U8:
        return CUDNN_DATA_UINT8;
    case Tensor::DataType::I32:
        return CUDNN_DATA_INT32;
    default:
        fprintf(stderr, "Unsupported data_type %d for cuda tensor\n",
                (int)data_type);
        abort();
    }
}

CudaTensor::CudaTensor(DataType data_type, const Dims &size,
                       cudnnTensorFormat_t format,
                       const std::shared_ptr<CudaContext> &ctx,
                       const std::optional<const std::string> &name)
  : Tensor(data_type, size, name)
  , m_type(cudnnDataType_from_dataType(data_type))
  , m_offset(0)
  , m_partial(false)
{
    chkCUDNN(cudnnCreateTensorDescriptor(&m_desc));
    assert(size.size() >= 0 && size.size() <= 4);
    chkCUDNN(cudnnSetTensor4dDescriptor(m_desc, format, m_type, size[0],
                                        size.size() > 1 ? (int)size[1] : 1,
                                        size.size() > 2 ? (int)size[2] : 1,
                                        size.size() > 3 ? (int)size[3] : 1));

    size_t bytes;
    chkCUDNN(cudnnGetTensorSizeInBytes(m_desc, &bytes));

    m_storage = std::make_shared<CudaTensorStorage>(data_type, bytes, ctx);
}

CudaTensor::CudaTensor(std::shared_ptr<CudaTensorStorage> storage,
                       const Dims &size, cudnnTensorFormat_t format,
                       const std::optional<const std::string> &name)
  : Tensor(storage->m_data_type, size, name)
  , m_type(cudnnDataType_from_dataType(storage->m_data_type))
  , m_offset(0)
  , m_partial(false)
{
    chkCUDNN(cudnnCreateTensorDescriptor(&m_desc));
    assert(size.size() >= 0 && size.size() <= 4);
    chkCUDNN(cudnnSetTensor4dDescriptor(m_desc, format, m_type, size[0],
                                        size.size() > 1 ? (int)size[1] : 1,
                                        size.size() > 2 ? (int)size[2] : 1,
                                        size.size() > 3 ? (int)size[3] : 1));
    m_storage = storage;
}

CudaTensor::CudaTensor(std::shared_ptr<CudaTensor> alias, const Dims &size,
                       std::vector<int64_t> offset_element,
                       const std::optional<const std::string> &name)
  : Tensor(alias->m_storage->m_data_type, size, name)
  , m_type(cudnnDataType_from_dataType(alias->m_storage->m_data_type))
  , m_offset(alias->m_offset)
  , m_storage(alias->m_storage)
  , m_partial(true)
{
    const int max_rank = 8;
    int dimsA[max_rank];
    int stridesA[max_rank];
    int rank;
    cudnnDataType_t data_type;

    chkCUDNN(cudnnGetTensorNdDescriptor(alias->m_desc, max_rank, &data_type,
                                        &rank, dimsA, stridesA));

    assert(data_type == m_type);
    for(size_t i = 0; i < size.size(); i++) {
        dimsA[i] = size[i];
    }

    chkCUDNN(cudnnCreateTensorDescriptor(&m_desc));
    chkCUDNN(cudnnSetTensorNdDescriptor(m_desc, m_type, rank, dimsA, stridesA));

    for(size_t i = 0; i < size.size() && i < offset_element.size(); i++) {
        m_offset += offset_element[i] * stridesA[i];
    }
}

CudaTensor::CudaTensor(std::shared_ptr<CudaTensorStorage> storage,
                       const Dims &size, int64_t offset, const int *strides,
                       const std::optional<const std::string> &name)
  : Tensor(storage->m_data_type, size, name)
  , m_type(cudnnDataType_from_dataType(storage->m_data_type))
  , m_offset(offset)
  , m_storage(storage)
  , m_partial(true)
{
    chkCUDNN(cudnnCreateTensorDescriptor(&m_desc));
    chkCUDNN(cudnnSetTensorNdDescriptor(m_desc, m_type, size.size(),
                                        &size.i32()[0], strides));
}

CudaTensor::CudaTensor(DataType data_type, const Dims &size, const int *strides,
                       const std::shared_ptr<CudaContext> &ctx,
                       const std::optional<const std::string> &name)
  : Tensor(data_type, size, name)
  , m_type(cudnnDataType_from_dataType(data_type))
  , m_partial(false)
{
    chkCUDNN(cudnnCreateTensorDescriptor(&m_desc));
    chkCUDNN(cudnnSetTensorNdDescriptor(m_desc, m_type, size.size(),
                                        &size.i32()[0], strides));
    size_t bytes;
    chkCUDNN(cudnnGetTensorSizeInBytes(m_desc, &bytes));

    m_storage = std::make_shared<CudaTensorStorage>(data_type, bytes, ctx);
}

CudaTensor::CudaTensor(const CudaTensor &o, cudnnTensorFormat_t format,
                       const std::optional<const std::string> &postfix)
  : CudaTensor(o.data_type_, o.dims_, format, o.m_storage->m_ctx,
               postfix ? o.namePostfix(*postfix) : std::nullopt)
{
}

CudaTensor::CudaTensor(const CudaTensor &o,
                       const std::optional<const std::string> &name)
  : Tensor(o.data_type_, o.dims_, name)
  , m_type(cudnnDataType_from_dataType(data_type_))
  , m_offset(0)
  , m_partial(false)
{
    const int max_rank = 8;
    int dimsA[max_rank];
    int stridesA[max_rank];
    int rank;
    cudnnDataType_t data_type;

    chkCUDNN(cudnnGetTensorNdDescriptor(o.m_desc, max_rank, &data_type, &rank,
                                        dimsA, stridesA));

    chkCUDNN(cudnnCreateTensorDescriptor(&m_desc));
    chkCUDNN(cudnnSetTensorNdDescriptor(m_desc, m_type, rank, dimsA, stridesA));
    size_t bytes;
    chkCUDNN(cudnnGetTensorSizeInBytes(m_desc, &bytes));

    m_storage = std::make_shared<CudaTensorStorage>(data_type_, bytes,
                                                    o.m_storage->m_ctx);
}

CudaTensor::CudaTensor(DataType data_type, const CudaTensor &o,
                       const std::optional<const std::string> &name)
  : Tensor(data_type, o.dims_, name)
  , m_type(cudnnDataType_from_dataType(data_type))
  , m_offset(0)
  , m_partial(false)
{
    const int max_rank = 8;
    int dimsA[max_rank];
    int stridesA[max_rank];
    int rank;
    cudnnDataType_t data_type_o;

    chkCUDNN(cudnnGetTensorNdDescriptor(o.m_desc, max_rank, &data_type_o, &rank,
                                        dimsA, stridesA));

    chkCUDNN(cudnnCreateTensorDescriptor(&m_desc));
    chkCUDNN(cudnnSetTensorNdDescriptor(m_desc, m_type, rank, dimsA, stridesA));
    size_t bytes;
    chkCUDNN(cudnnGetTensorSizeInBytes(m_desc, &bytes));

    m_storage = std::make_shared<CudaTensorStorage>(data_type_, bytes,
                                                    o.m_storage->m_ctx);
}

CudaTensor::~CudaTensor() { chkCUDNN(cudnnDestroyTensorDescriptor(m_desc)); }

std::unique_ptr<TensorAccess>
CudaTensor::access()
{
    if(!m_storage->data()) {
        m_storage->alloc();
    }
    return std::make_unique<CudaTensorAccess>(m_storage, m_desc, m_offset);
}

std::shared_ptr<Tensor>
CudaTensor::slice(const Dims &offset, const Dims &size)
{
    const int max_rank = 8;
    int dimsA[max_rank];
    int stridesA[max_rank];
    int rank;
    cudnnDataType_t data_type;

    chkCUDNN(cudnnGetTensorNdDescriptor(m_desc, max_rank, &data_type, &rank,
                                        dimsA, stridesA));

    int64_t o = m_offset;

    for(int i = 0; i < rank && i < (ssize_t)offset.size(); i++) {
        o += offset[i] * stridesA[i];
    }

    return std::make_shared<CudaTensor>(m_storage, size, o, stridesA,
                                        namePostfix("slice"));
}

std::shared_ptr<CudaTensor>
CudaTensor::makeSharedGrad()
{
    if(!m_grad)
        m_grad = std::make_shared<CudaTensor>(*this, namePostfix("grad"));

    return m_grad;
}

std::shared_ptr<CudaTensor>
CudaTensor::makePrivateGrad()
{
    return std::make_shared<CudaTensor>(*this, namePostfix("private_grad"));
}

void *
CudaTensor::deviceMem() const
{
    return m_storage->deviceMem(m_offset);
};

std::string
CudaTensor::shortname() const
{
    std::stringstream ss;
    ss << "T" << m_storage->m_id;
    if(name_) {
        ss << "\"" << *name_ << "\"";
    }

    if(m_offset) {
        ss << "[" << m_offset << "]";
    }

    return ss.str();
}

std::string
CudaTensor::hashkey() const
{
    char buf[128];

    const int max_rank = 8;
    int dims[max_rank];
    int strides[max_rank];
    int rank;
    cudnnDataType_t data_type;

    chkCUDNN(cudnnGetTensorNdDescriptor(m_desc, max_rank, &data_type, &rank,
                                        dims, strides));
    const char *dt = NULL;
    switch(data_type) {
    case CUDNN_DATA_FLOAT:
        dt = "f";
        break;
    case CUDNN_DATA_HALF:
        dt = "h";
        break;
    case CUDNN_DATA_UINT8:
        dt = "u8";
        break;
    case CUDNN_DATA_INT32:
        dt = "i32";
        break;
    default:
        abort();
    }

    switch(rank) {
    case 1:
        snprintf(buf, sizeof(buf), "%d;%d;%s", dims[0], strides[0], dt);
        break;
    case 2:
        snprintf(buf, sizeof(buf), "%d.%d;%d.%d;%s", dims[0], dims[1],
                 strides[0], strides[1], dt);
        break;
    case 3:
        snprintf(buf, sizeof(buf), "%d.%d.%d;%d.%d.%d;%s", dims[0], dims[1],
                 dims[2], strides[0], strides[1], strides[2], dt);
        break;
    case 4:
        snprintf(buf, sizeof(buf), "%d.%d.%d.%d;%d.%d.%d.%d;%s", dims[0],
                 dims[1], dims[2], dims[3], strides[0], strides[1], strides[2],
                 strides[3], dt);
        break;
    default:
        abort();
    }
    return buf;
}

std::string
CudaTensor::info() const
{
    std::stringstream ss;
    ss << "T" << m_storage->m_id;
    if(name_)
        ss << "\"" << *name_ << "\"";

    const int max_rank = 8;
    int dims[max_rank];
    int strides[max_rank];
    int rank;
    cudnnDataType_t data_type;

    chkCUDNN(cudnnGetTensorNdDescriptor(m_desc, max_rank, &data_type, &rank,
                                        dims, strides));
    switch(data_type) {
    case CUDNN_DATA_FLOAT:
        ss << "<float>";
        break;
    case CUDNN_DATA_HALF:
        ss << "<half>";
        break;
    case CUDNN_DATA_UINT8:
        ss << "<u8>";
        break;
    case CUDNN_DATA_INT32:
        ss << "<i32>";
        break;
    default:
        ss << "<?>";
        break;
    }

    const char *prefix = "";
    ss << "[";
    for(int i = 0; i < rank; i++) {
        ss << prefix << dims[i];
        prefix = ", ";
    }
    ss << "]";

    prefix = "";
    ss << "{";
    for(int i = 0; i < rank; i++) {
        ss << prefix << strides[i];
        prefix = ", ";
    }
    ss << "}@cuda:" << m_storage->deviceMem(0);

    if(m_storage->m_mem) {
        ss << "<static>";
    }

    if(m_offset) {
        ss << " + " << m_offset;
    }

    if(m_partial) {
        ss << " <partial>";
    }

    if(m_storage->m_inf_is_valid) {
        ss << " <may-have-inf>";
    }

    return ss.str();
}

bool
CudaTensor::cpacked() const
{
    const int max_rank = 8;
    int dims[max_rank];
    int strides[max_rank];
    int rank;
    cudnnDataType_t data_type;

    chkCUDNN(cudnnGetTensorNdDescriptor(m_desc, max_rank, &data_type, &rank,
                                        dims, strides));

    return strides[1] == 1;
}

void
CudaTensor::copyFromLocked(Tensor &t, int dst_broadcast_dimension)
{
    const int max_rank = 8;
    int dims[max_rank];
    int strides[max_rank];
    int rank;
    cudnnDataType_t data_type;

    auto ta = t.access();
    if(ta == nullptr) {
        return;
    }

    m_storage->alloc();

    chkCUDNN(cudnnGetTensorNdDescriptor(m_desc, max_rank, &data_type, &rank,
                                        dims, strides));

    cudaStreamSynchronize(m_storage->m_ctx->m_stream);

    if(!copy_tensor(m_storage->deviceMem(m_offset), dims_.size(),
                    &dims_.i32()[0], &strides[0], data_type_, t, ta.get(),
                    dst_broadcast_dimension)) {
        fprintf(stderr,
                "Cuda Tensor copy failed\n"
                "From: %s\n"
                "  To: %s\n",
                t.info().c_str(), info().c_str());
        abort();
    }
}

Tensor::Stats
CudaTensor::stats()
{
    auto s = m_storage;
    if(s == nullptr)
        return Stats({0, 0, 0, 0});

    auto ctx = m_storage->m_ctx;

    ctx->m_workspace.alloc();

    float *output = (float *)ctx->m_workspace.ptr();
    const size_t elements = dims_.elements();

    switch(m_type) {
    case CUDNN_DATA_FLOAT:
        tensor_stats_float(elements, (const float *)deviceMem(), output,
                           ctx->m_stream);
        break;
    case CUDNN_DATA_HALF:
        tensor_stats_half(elements, (const __half *)deviceMem(), output,
                          ctx->m_stream);
        break;
    default:
        return Tensor::stats();
    }

    cudaStreamSynchronize(ctx->m_stream);

    const float min = output[0];
    const float max = output[1];
    const float mean = output[2];
    const float var = output[3];

    return Stats({.min = min, .max = max, .mean = mean, .stddev = sqrtf(var)});
}

struct CollapseDim {
    int size;
    int stride;
    int index;
};

void
CudaTensor::detect_anomaly(uint32_t *ptr)
{
    auto s = m_storage;
    if(s == nullptr)
        return;

    const int max_rank = 8;
    cudnnDataType_t data_type;
    int rank;
    int dimsA[max_rank];
    int stridesA[max_rank];

    chkCUDNN(cudnnGetTensorNdDescriptor(m_desc, max_rank, &data_type, &rank,
                                        dimsA, stridesA));
    std::vector<CollapseDim> dv;
    for(int i = 0; i < rank; i++) {
        dv.push_back({dimsA[i], stridesA[i], i});
    }

    std::sort(dv.begin(), dv.end(),
              [](const CollapseDim &a, const CollapseDim &b) {
                  if(a.stride == b.stride)
                      return a.size > b.size;
                  return a.stride > b.stride;
              });

    for(int i = rank - 1; i > 0; i--) {
        if(dv[i].size * dv[i].stride == dv[i - 1].stride) {
            dv[i - 1].size *= dv[i].size;
            dv[i - 1].stride = dv[i].stride;
            dv.erase(dv.begin() + i);
        }
    }

    if(dv.size() == 1) {
        switch(data_type) {
        case CUDNN_DATA_FLOAT:
            find_non_finite_float_1d(dv[0].size, (const float *)deviceMem(),
                                     ptr, true, s->m_ctx->m_stream);
            break;
        case CUDNN_DATA_HALF:
            find_non_finite_half_1d(dv[0].size, (const __half *)deviceMem(),
                                    ptr, !s->m_inf_is_valid,
                                    s->m_ctx->m_stream);
            break;
        default:
            abort();
        }
    } else if(dv.size() == 2) {
        switch(data_type) {
        case CUDNN_DATA_FLOAT:
            find_non_finite_float_2d(dv[1].size, dv[0].size, dv[0].stride,
                                     (const float *)deviceMem(), ptr, true,
                                     s->m_ctx->m_stream);
            break;
        case CUDNN_DATA_HALF:
            find_non_finite_half_2d(dv[1].size, dv[0].size, dv[0].stride,
                                    (const __half *)deviceMem(), ptr,
                                    !s->m_inf_is_valid, s->m_ctx->m_stream);
            break;
        default:
            abort();
        }
    } else {
        fprintf(stderr,
                "CudaTensor::detect_anomaly(): Dimensionality of %zd not "
                "supported\n",
                dv.size());
        abort();
    }
}

size_t
size_from_params(const Dims &size, cudnnTensorFormat_t format,
                 cudnnDataType_t data_type)
{
    cudnnTensorDescriptor_t desc;
    chkCUDNN(cudnnCreateTensorDescriptor(&desc));
    assert(size.size() >= 0 && size.size() <= 4);

    chkCUDNN(cudnnSetTensor4dDescriptor(desc, format, data_type, size[0],
                                        size.size() > 1 ? (int)size[1] : 1,
                                        size.size() > 2 ? (int)size[2] : 1,
                                        size.size() > 3 ? (int)size[3] : 1));

    size_t bytes;
    chkCUDNN(cudnnGetTensorSizeInBytes(desc, &bytes));

    cudnnDestroyTensorDescriptor(desc);
    return bytes;
}

CudaTensorStorageDoubleBuffered::CudaTensorStorageDoubleBuffered(
    Tensor::DataType data_type, Dims &dims, cudnnTensorFormat_t format,
    const std::shared_ptr<CudaContext> &ctx)
  : CudaTensorStorage(
        data_type,
        size_from_params(dims, format, cudnnDataType_from_dataType(data_type)),
        ctx)
  , m_index(0)
{
    for(int i = 0; i < 2; i++) {
        chkCuda(cudaMallocManaged(&m_buffers[i], m_size, cudaMemAttachGlobal));
        chkCuda(cudaMemset(m_buffers[i], 0, m_size));
    }
}

CudaTensorStorageDoubleBuffered::~CudaTensorStorageDoubleBuffered()
{
    for(int i = 0; i < 2; i++) {
        chkCuda(cudaFree(m_buffers[i]));
    }
}

void *
CudaTensorStorageDoubleBuffered::deviceMem(int64_t offset)
{
    void *buf = m_buffers[m_index & 1];

    void *r = (void *)((char *)buf + offset * m_element_size);
    return r;
}

void *
CudaTensorStorageDoubleBuffered::data(int buffer) const
{
    return m_buffers[(buffer + m_index) & 1];
}

double
CudaTensorStorageDoubleBuffered::get(size_t offset) const
{
    return m_get(data(0), offset);
}

double
CudaTensorStorageDoubleBuffered::get(size_t offset, int buffer) const
{
    return m_get(data(buffer), offset);
}

void
CudaTensorStorageDoubleBuffered::set(size_t offset, double value)
{
    m_set(data(0), offset, value);
}

void
CudaTensorStorageDoubleBuffered::set(size_t offset, double value, int buffer)
{
    m_set(data(buffer), offset, value);
}

int
CudaTensorStorageDoubleBuffered::flip()
{
    m_index++;
    return m_index & 1;
}

void
CudaTensorStorageDoubleBuffered::prefetchGPU()
{
    cudaMemPrefetchAsync(data(1), m_size, m_ctx->m_deviceId, m_ctx->m_stream);
}

class CudaTensorBatchAccess : public TensorAccess {
    static const int MAX_RANK = 8;

public:
    CudaTensorBatchAccess(CudaTensorStorageDoubleBuffered *storage,
                          cudnnTensorDescriptor_t desc, int64_t offset)
      : m_storage(storage), m_offset(offset)
    {
        int dims[MAX_RANK];
        cudnnDataType_t data_type;

        chkCUDNN(cudnnGetTensorNdDescriptor(desc, MAX_RANK, &data_type, &m_rank,
                                            dims, m_strides));
    }

    int64_t offsetForElement(const Dims &element) const
    {
        size_t offset = m_offset;
        for(int i = 0; i < (int)element.size() && i < m_rank; i++) {
            offset += element[i] * m_strides[i];
        }
        return offset;
    }

    Dims strides() override { abort(); }

    void *data() override { abort(); };

    void copyBytesFrom(const Dims &element, const void *data,
                       size_t size) override
    {
        const size_t o = offsetForElement(element) * m_storage->m_element_size;
        char *dst = (char *)m_storage->data(1);
        cudaMemcpy(dst + o, data, size, cudaMemcpyHostToDevice);
    }

    void *getAddr(const Dims &element) override
    {
        size_t off = offsetForElement(element) * m_storage->m_element_size;
        return (void *)((char *)m_storage->data(1) + off);
    };

    double get(const Dims &element) override
    {
        return m_storage->get(offsetForElement(element), 1);
    };

    void set(const Dims &element, double value) override
    {
        m_storage->set(offsetForElement(element), value, 1);
    }

    CudaTensorStorageDoubleBuffered *m_storage;
    int64_t m_offset;
    int m_rank;
    int m_strides[MAX_RANK];
};

void
CudaProgram::run_batched_tensor_callbacks(const TensorBatchCallback &cb,
                                          bool training, long batch,
                                          const CudaBatchAccessOps &list)
{
    if(!cb)
        return;

#ifdef __aarch64__
    cudaStreamSynchronize(ctx_->m_stream);
#endif

    std::unordered_map<std::shared_ptr<Tensor>, TensorAccess *> amap;

    std::vector<std::unique_ptr<CudaTensorBatchAccess>> v;
    for(auto &op : list) {
        auto ta = std::make_unique<CudaTensorBatchAccess>(
            op.m_storage.get(), op.m_low->m_desc, op.m_low->m_offset);
        amap[op.m_high] = ta.get();
        v.push_back(std::move(ta));
    }
    cb(batch, training, amap);
}

void
CudaProgram::flipDoubleBufferedTensors()
{
    for(const auto &s : m_flips) {
        s->flip();
    }
}

}  // namespace saga
