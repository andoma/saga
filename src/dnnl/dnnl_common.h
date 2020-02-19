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

#pragma once

#include "saga.h"
#include "dnnl.h"
#include "dnnl_debug.h"

#define chkDNNL(f)                                              \
  do {                                                          \
    dnnl_status_t s_ = f;                                       \
    if (s_ != dnnl_success) {                                   \
      fprintf(stderr, "DNNL error at %s:%d in %s: %s\n",        \
              __FILE__, __LINE__, __FUNCTION__,                 \
              dnnl_status2str(s_));                             \
      abort();                                                  \
    }                                                           \
  } while (0)


namespace saga {

class DnnlContext : public Context,
                    public std::enable_shared_from_this<DnnlContext> {

public:

  DnnlContext();
  ~DnnlContext();

  std::shared_ptr<Program> createProgram(const Graph &graph,
                                         const ProgramConfig &pc);


  dnnl_engine_t engine_;
  dnnl_stream_t stream_;
};

}
