#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "saga.h"

namespace saga {


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
