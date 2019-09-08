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


  const float *src0 = (const float *)host_mem_;

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



void Tensor::savePng(const char *filename, int num, int channel) const
{
  int nn, cc, hh, ww, ns, cs, hs, ws;
  cudnnDataType_t datatype;
  cudnnGetTensor4dDescriptor(desc(), &datatype,
                             &nn, &cc, &hh, &ww,
                             &ns, &cs, &hs, &ws);

  const int nn_count = num     == -1 ? nn : 1;
  const int cc_count = channel == -1 ? cc : 1;

  const int nn_offset = num     == -1 ? 0 : num;
  const int cc_offset = channel == -1 ? 0 : channel;

  printf("%d %d %d %d  strides: %d %d %d %d\n",
         nn,cc,hh,ww, ns, cs, hs, ws);

  assert(datatype == CUDNN_DATA_FLOAT);

  const int one_size = cc_count * hh * ww;
  const int total_size = nn_count * one_size;

  float *copy = (float *)malloc(bytes_);
  cudaMemcpy((void *)copy, device_mem_,
             bytes_, cudaMemcpyDeviceToHost);

  uint8_t *imgbuf = (uint8_t *)malloc(total_size);

  uint8_t *dst = imgbuf;

  for(int i = 0; i < nn_count; i++) {
    for(int y = 0; y < hh; y++) {
      for(int x = 0; x < ww; x++) {
        for(int z = 0; z < cc_count; z++) {
          const float *src = copy +
            (nn_offset + i) * ns + y * hs + x * ws + (cc_offset + z) * cs;
          const float v = *src;
          *dst++ = std::max(0.0f, std::min(1.0f, v * 0.5f + 0.5f)) * 255;
        }
      }
    }
  }

  printf("Writing PNG %s width:%d height:%d channels:%d stride:%d\n",
         filename, ww, hh * nn_count, cc_count, ww * cc_count);


  stbi_write_png(filename, ww, hh * nn_count, cc_count, (void *)imgbuf,
                 ww * cc_count);

  free(imgbuf);
  free(copy);
}

}
