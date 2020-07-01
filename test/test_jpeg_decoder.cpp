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




static void
fill_theta(Tensor *t, int batch_size)
{
  auto ta = t->access();
  for(int i = 0; i < batch_size; i++) {

    ta->set({i, 1, 0}, -1);
    ta->set({i, 0, 1}, 1);
  }
}



static int
jpeg_decoder_main(int argc, char **argv)
{
  int opt;
  auto dt = Tensor::DataType::FLOAT;

  int transform = 0;
  const int batch_size = 4;

  while((opt = getopt(argc, argv, "hvt")) != -1) {
    switch(opt) {
    case 'h':
      dt = Tensor::DataType::HALF;
      break;
    case 'v':
      g_verbose++;
      break;
    case 't':
      transform = 1;
      break;
    }
  }

  argc -= optind;
  argv += optind;

  Graph g;

  auto n = g.addJpegDecoder(32, 32, dt,
                            [&](long batch, int n, uint8_t *data, size_t len) {
                              return load_jpeg(batch, n, data, len);
                            });

  std::shared_ptr<Tensor> theta =
    makeCPUTensor(Tensor::DataType::FLOAT,
                  Dims({batch_size, 2, 3}), "theta0");

  if(transform)
    n = g.addSpatialTransform(n->y(), theta, -1, -1, true);

  if(g_verbose)
    g.print();

  auto ctx = createContext();
  auto p = ctx->createProgram(g, {
      .inference = true,
      .training = false,
      .batch_size = batch_size
   });



  if(g_verbose > 1)
    p->print();

  theta = p->resolveTensor(theta);
  if(theta)
    fill_theta(theta.get(), batch_size);

  auto output = p->resolveTensor(n->y());

  p->infer(2);

  output->printRGB("GBRG");

  return 0;
}


SAGA_CLI_CMD("jpeg-decoder",
             "jpeg-decoder [OPTIONS ...]",
             "Run JPEG decoder test",
             jpeg_decoder_main);
