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

  synchronize();
  const float *src0 = (const float *)deviceMem();

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
