// -*-c++-*-

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

#include "saga.h"

#include "dnnl_common.h"

namespace saga {

class DnnlTensorStorage;


class DnnlTensor : public Tensor {

public:
  DnnlTensor(const dnnl_memory_desc_t *desc,
             const std::shared_ptr<DnnlContext> &ctx,
             const std::optional<const std::string> &name = std::nullopt);

  ~DnnlTensor();

  virtual std::string info() const override;

  std::unique_ptr<TensorAccess> access() override;

  std::shared_ptr<Tensor> slice(const Dims &offset, const Dims &size) override;

  std::shared_ptr<Tensor> grad() const override {
    return grad_;
  }

  void copyFrom(Tensor &t) override;

  void *deviceMem() const;

  std::shared_ptr<DnnlTensor> makeGrad();

  void copyFromLocked(Tensor &t);

  const dnnl_memory_desc_t desc_;
  dnnl_memory_t memory_;

  std::shared_ptr<DnnlTensorStorage> storage_;
  std::shared_ptr<DnnlTensor> grad_;
};

}
