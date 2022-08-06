// -*-c++-*-

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

#include "saga.hpp"

#include "cuda_common.hpp"

namespace saga {

class CudaTensorStorage : public TensorStorage {
public:
    CudaTensorStorage(Tensor::DataType data_type, size_t size,
                      const std::shared_ptr<CudaContext> &ctx);

    virtual ~CudaTensorStorage();

    void alloc();
    void setTmpMem(void *p);

    virtual void *deviceMem(int64_t offset);

    const std::shared_ptr<CudaContext> m_ctx;
    const size_t m_size;
    const size_t m_element_size;
    const int m_id;

    void *m_mem = nullptr;

    bool m_nonfinite_is_valid{false};

    void invalidate();
};

class CudaTensorStorageDoubleBuffered : public CudaTensorStorage {
public:
    CudaTensorStorageDoubleBuffered(Tensor::DataType data_type, Dims &dims,
                                    cudnnTensorFormat_t format,
                                    const std::shared_ptr<CudaContext> &ctx);

    virtual ~CudaTensorStorageDoubleBuffered();

    void *deviceMem(int64_t offset) override;
    void *data(int buffer) const;

    double get(size_t offset) const override;
    void set(size_t offset, double value) override;

    double get(size_t offset, int buffer) const;
    void set(size_t offset, double value, int buffer);
    int flip();
    void prefetchGPU();

    void *m_buffers[2];
    int m_index;
};

class CudaTensor : public Tensor {
public:
    CudaTensor(DataType data_type, const Dims &size, cudnnTensorFormat_t format,
               const std::shared_ptr<CudaContext> &ctx,
               const std::optional<const std::string> &name = std::nullopt);

    CudaTensor(std::shared_ptr<CudaTensorStorage> storage, const Dims &size,
               cudnnTensorFormat_t format,
               const std::optional<const std::string> &name = std::nullopt);

    CudaTensor(std::shared_ptr<CudaTensor> alias, const Dims &size,
               std::vector<int64_t> offset_element,
               const std::optional<const std::string> &name = std::nullopt);

    CudaTensor(std::shared_ptr<CudaTensorStorage> storage, const Dims &size,
               int64_t offset, const int *strides,
               const std::optional<const std::string> &name = std::nullopt);

    CudaTensor(DataType data_type, const Dims &size, const int *strides,
               const std::shared_ptr<CudaContext> &ctx,
               const std::optional<const std::string> &name);

    CudaTensor(
        const CudaTensor &t, cudnnTensorFormat_t format,
        const std::optional<const std::string> &name_postfix = std::nullopt);

    CudaTensor(const CudaTensor &o,
               const std::optional<const std::string> &name);

    CudaTensor(DataType data_type, const CudaTensor &o,
               const std::optional<const std::string> &name = std::nullopt);

    ~CudaTensor();

    virtual std::string info() const override;

    std::unique_ptr<TensorAccess> access() override;

    std::shared_ptr<Tensor> slice(const Dims &offset,
                                  const Dims &size) override;

    Stats stats() override;

    cudnnTensorDescriptor_t desc() const { return m_desc; }

    void *deviceMem() const;

    std::shared_ptr<CudaTensor> makeSharedGrad();

    std::shared_ptr<CudaTensor> makePrivateGrad();

    bool cpacked() const;

    void copyFromLocked(Tensor &t, int dst_broadcast_dimension = -1);

    int storage_id() const { return m_storage->m_id; }

    std::string shortname() const;

    size_t memoryUsage() const { return m_storage->m_size; }

    std::string hashkey() const;

    void detect_anomaly(uint32_t *ptr);

    void invalidate(void);

    std::vector<std::pair<int64_t, int64_t>> collapse_dims(void) const;

    const cudnnDataType_t m_type;
    int64_t m_offset;
    std::shared_ptr<CudaTensorStorage> m_storage;
    cudnnTensorDescriptor_t m_desc;
    std::shared_ptr<CudaTensor> m_grad;
    bool m_partial{false};
};

}  // namespace saga
