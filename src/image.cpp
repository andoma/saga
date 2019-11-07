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

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "saga.h"

#include "turbo_colormap.h"

namespace saga {


void Tensor::toRGBBitmap(uint8_t *output, int stride,
                         int nn_offset, int channel,
                         float min, float max) const
{
  int nn, cc, hh, ww, ns, cs, hs, ws;
  cudnnDataType_t datatype;
  cudnnGetTensor4dDescriptor(desc(), &datatype,
                             &nn, &cc, &hh, &ww,
                             &ns, &cs, &hs, &ws);

  const int cc_count  = channel == -1 ? std::min(cc, 3) : 1;
  const int cc_offset = channel == -1 ? 0               : channel;

  assert(datatype == CUDNN_DATA_FLOAT);

  const float *src0 = (const float *)hostMem();

  for(int y = 0; y < hh; y++) {
    for(int x = 0; x < ww; x++) {
      for(int z = 0; z < cc_count; z++) {
        const float *src = src0 +
          nn_offset * ns + y * hs + x * ws + (cc_offset + z) * cs;
        const float v = *src;
        min = std::min(v, min);
        max = std::max(v, max);
      }
    }
  }

  const float s = 1.0f / (max - min);
  const float k = -min;

  if(channel != -1) {
    for(int y = 0; y < hh; y++) {
      uint8_t *dst = output + y * stride;
      for(int x = 0; x < ww; x++) {

        const float *src = src0 +
          nn_offset * ns + y * hs + x * ws + (cc_offset) * cs;
        const float v = (*src + k) * s;
        uint8_t idx = v * 255;
        *dst++ = turbo_srgb_bytes[idx][0];
        *dst++ = turbo_srgb_bytes[idx][1];
        *dst++ = turbo_srgb_bytes[idx][2];
      }
    }
  } else {
    for(int y = 0; y < hh; y++) {
      uint8_t *dst = output + y * stride;
      for(int x = 0; x < ww; x++) {
        int z = 0;
        for(z = 0; z < cc_count; z++) {
          const float *src = src0 +
            nn_offset * ns + y * hs + x * ws + (cc_offset + z) * cs;
          const float v = (*src + k) * s;
          *dst++ = v * 255;
        }
        for(; z < 3; z++) {
          *dst++ = 0;
        }
      }
    }
  }
}


}
