#include <signal.h>
#include <assert.h>
#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>

#include <algorithm>
#include <numeric>
#include <string.h>

#include "saga.h"
#include "cli.h"

using namespace saga;

static int g_verbose = 0;

#include "test_jpegs.h"


static size_t
load_jpeg(long batch, int n, uint8_t *data, size_t len)
{
  const uint8_t *src;
  size_t srclen;
  n += batch;
  if((n % 3) == 0) {
    src = jpeg_r;
    srclen = sizeof(jpeg_r);
  } else if((n % 3) == 1) {
    src = jpeg_g;
    srclen = sizeof(jpeg_g);
  } else {
    src = jpeg_b;
    srclen = sizeof(jpeg_b);
  }
  memcpy(data, src, srclen);
  return srclen;
}

static int
jpeg_decoder_main(int argc, char **argv)
{
  int opt;
  auto dt = Tensor::DataType::FLOAT;

  while((opt = getopt(argc, argv, "hv")) != -1) {
    switch(opt) {
    case 'h':
      dt = Tensor::DataType::HALF;
      break;
    case 'v':
      g_verbose++;
      break;
    }
  }

  argc -= optind;
  argv += optind;

  printf("dt=%d\n", (int)dt);

  Graph g;

  auto n =  g.addNode("jpegdecoder", [&](long batch, int n,
                                         uint8_t *data, size_t len)
                      {
                        return load_jpeg(batch, n, data, len);
                      }, {{"width", 32}, {"height", 32}});

  g.print();

  auto ctx = createContext();
  auto p = ctx->createProgram(g, {
      .inference = true,
      .training = false,
      .batch_size = 4
   });

  p->print();

  auto output = p->resolveTensor(n->y());

  p->infer(2);

  output->printRGB("GBRG");

  return 0;
}


SAGA_CLI_CMD("jpeg-decoder",
             "jpeg-decoder [OPTIONS ...]",
             "Run JPEG decoder test",
             jpeg_decoder_main);
